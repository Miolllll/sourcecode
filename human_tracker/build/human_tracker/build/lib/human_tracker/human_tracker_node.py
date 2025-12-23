#!/usr/bin/env python3
import os
import time
import math
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ------------ Simple structure ------------
@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int,int,int,int]   # (x, y, w, h) in raw image
    center: Tuple[float,float]     # (cx, cy) in raw image
    depth_m: float                 # meters


# --------------- Main Node ---------------
class HumanTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')

        # ---- Parameters (only essentials) ----
        self.declare_parameter('rgb_topic',   '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('caminfo_topic', '/k4a/rgb/camera_info')

        self.declare_parameter('yolo_weight', 'yolo11n.pt')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf_thr', 0.40)
        self.declare_parameter('iou_thr', 0.45)

        # show window (set false on headless)
        self.declare_parameter('show_window', True)

        # depth ROI / validity
        self.declare_parameter('depth_roi_frac', 0.35)        # fraction of bbox (each axis)
        self.declare_parameter('depth_valid_ratio_min', 0.25) # min valid px ratio in ROI
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 8.0)

        # keep at most two people on screen
        self.max_people = 2

        # ---- Load params ----
        self.rgb_topic   = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.caminfo_topic = self.get_parameter('caminfo_topic').value

        self.weight_name = self.get_parameter('yolo_weight').value
        self.imgsz       = int(self.get_parameter('imgsz').value)
        self.conf_thr    = float(self.get_parameter('conf_thr').value)
        self.iou_thr     = float(self.get_parameter('iou_thr').value)

        self.show_window = bool(self.get_parameter('show_window').value)

        self.depth_roi_frac = float(self.get_parameter('depth_roi_frac').value)
        self.depth_valid_ratio_min = float(self.get_parameter('depth_valid_ratio_min').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        # ---- State ----
        self.bridge = CvBridge()
        self.depth_frame = None
        self.fx = None; self.fy = None; self.cx_cam = None; self.cy_cam = None

        self.primary_id = None  # PointStamped로 publish 하는 고유 사용자
        self.last_goal  = None  # 마지막 publish 값 (추정 유지용)

        # ---- Device / perf ----
        self.gpu = torch.cuda.is_available()
        self.device_str = 'cuda:0' if self.gpu else 'cpu'
        torch.backends.cudnn.benchmark = True

        # ---- YOLO load (person class only) ----
        model_file = os.path.join(os.path.dirname(__file__), self.weight_name)
        if not os.path.exists(model_file):
            self.get_logger().warn(f"Downloading {self.weight_name} …")
            urllib.request.urlretrieve(
                f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{self.weight_name}",
                model_file
            )
        self.yolo = YOLO(model_file)
        try:
            self.yolo.to(self.device_str)
            self.yolo.fuse()
        except Exception as e:
            self.get_logger().warn(f"YOLO move/fuse failed: {e}")

        # ---- DeepSORT (light config) ----
        self.tracker = DeepSort(
            max_age=20, n_init=3,
            max_cosine_distance=0.30,
            nn_budget=100,
            embedder='mobilenet',  # 내부 경량 임베더 사용
            embedder_gpu=self.gpu,
            half=True if self.gpu else False,
            bgr=True,
        )

        # ---- ROS I/O ----
        self.create_subscription(Image, self.rgb_topic,   self.rgb_cb,   qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.caminfo_topic, self.caminfo_cb, qos_profile_sensor_data)
        self.user_pub = self.create_publisher(PointStamped, '/user_tracking', 10)

        # UI
        if self.show_window:
            try:
                cv2.namedWindow("Human Tracker", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Human Tracker", 960, 540)
            except Exception as e:
                self.get_logger().warn(f"OpenCV window init failed: {e}")

        self.get_logger().info("✅ HumanTracker (lightweight) initialized")

    # ---------- Callbacks ----------
    def depth_cb(self, msg: Image):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"[depth_cb] convert failed: {e}")

    def caminfo_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx_cam = msg.k[2]; self.cy_cam = msg.k[5]

    # ---------- Helpers ----------
    def _depth_from_bbox(self, bbox, depth_img) -> float:
        """Compute median depth (m) inside a small ROI around bbox center."""
        x,y,w,h = map(int, bbox)
        H, W = depth_img.shape[:2]
        cx, cy = int(x + w/2), int(y + h/2)

        rw = max(8, int(w * self.depth_roi_frac))
        rh = max(8, int(h * self.depth_roi_frac))

        x0 = max(0, cx - rw//2); x1 = min(W, cx + rw//2)
        y0 = max(0, cy - rh//2); y1 = min(H, cy + rh//2)

        roi = depth_img[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        valid = roi[roi > 0]
        if valid.size / float(roi.size) < self.depth_valid_ratio_min:
            return None

        scale = 0.001 if roi.dtype == np.uint16 else 1.0
        d = float(np.median(valid)) * scale
        if not (self.min_depth_m <= d <= self.max_depth_m):
            return None
        return d

    def _project_to_camera(self, cx, cy, z_m):
        """Map pixel center + depth to simple camera frame (same convention as 기존 코드)."""
        X = (cx - self.cx_cam) * z_m / self.fx
        Y = (cy - self.cy_cam) * z_m / self.fy
        # camera_base 좌표계로 맞춤 (기존 노드와 동일)
        px = float(z_m)
        py = float(-X)
        pz = float(-Y)
        return px, py, pz

    # ---------- Main RGB ----------
    def rgb_cb(self, msg: Image):
        try:
            if self.depth_frame is None or self.fx is None:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            H, W = frame.shape[:2]

            # YOLO inference (person only)
            with torch.inference_mode():
                res = self.yolo.predict(
                    frame, classes=[0], imgsz=self.imgsz,
                    device=(0 if self.gpu else 'cpu'),
                    half=self.gpu, conf=self.conf_thr,
                    iou=self.iou_thr, verbose=False
                )[0]

            # Parse detections for DeepSORT
            dets_ds = []
            frame_area = float(W * H)
            min_area = 0.001 * frame_area  # drop tiny boxes
            for b in res.boxes:
                conf = float(b.conf[0])
                if conf < self.conf_thr:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].detach().cpu().numpy())
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
                w, h = x2-x1, y2-y1
                if w*h < min_area or w <= 1 or h <= 1:
                    continue
                dets_ds.append(((x1, y1, w, h), conf, None))

            # Tracking
            try:
                tracks = self.tracker.update_tracks(dets_ds, frame=frame)
            except Exception as e:
                self.get_logger().error(f"[DeepSort] update failed: {e}")
                return

            # Build detections with depth
            dets: List[Detection] = []
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                x1,y1,x2,y2 = map(int, tr.to_tlbr())
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                cx, cy = x1 + w/2.0, y1 + h/2.0
                d = self._depth_from_bbox((x1, y1, w, h), self.depth_frame)
                if d is None:
                    continue
                dets.append(Detection(tr.track_id, (x1,y1,w,h), (cx,cy), d))

            if not dets:
                # nothing to draw/publish; keep last goal published for continuity
                if self.show_window:
                    cv2.imshow('Human Tracker', frame)
                    cv2.waitKey(1)
                return

            # Limit to at most two people (closest by depth)
            dets.sort(key=lambda z: z.depth_m)
            dets = dets[:self.max_people]

            # Decide / keep primary (PointStamped person = RED)
            ids_present = {d.track_id for d in dets}
            if self.primary_id not in ids_present:
                # pick the closest one
                self.primary_id = dets[0].track_id

            # Draw & publish
            for d in dets:
                l, t, w, h = d.bbox
                r, b = l + w, t + h
                is_primary = (d.track_id == self.primary_id)

                # publish for primary
                if is_primary:
                    px, py, pz = self._project_to_camera(d.center[0], d.center[1], d.depth_m)
                    pt = PointStamped()
                    pt.header.stamp = msg.header.stamp
                    pt.header.frame_id = 'camera_base'
                    pt.point.x, pt.point.y, pt.point.z = px, py, pz
                    self.user_pub.publish(pt)
                    self.last_goal = pt

                color = (0,0,255) if is_primary else (0,255,0)  # red for primary, green otherwise
                cv2.rectangle(frame, (l,t), (r,b), color, 2)
                label = ("USER" if is_primary else "ID") + f"{d.track_id} {d.depth_m:.2f}m"
                cv2.putText(frame, label, (l, max(20, t-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if self.show_window:
                cv2.imshow('Human Tracker', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"[rgb_cb] exception: {e}")

    # ---------- Shutdown ----------
    def destroy_node(self):
        try:
            if self.show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HumanTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
