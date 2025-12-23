#!/usr/bin/env python3
import os
import time
import math
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

        # [중요] 정확한 모델명으로 수정: yolo11s.pt
        self.declare_parameter('yolo_weight', 'yolov11s.pt')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf_thr', 0.60)
        self.declare_parameter('iou_thr', 0.45)

        self.declare_parameter('show_window', True)

        self.declare_parameter('depth_roi_frac', 0.35)
        self.declare_parameter('depth_valid_ratio_min', 0.25)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 8.0)
        
        # [수정] 추적 안정성 관련 파라미터 추가
        # primary_id를 잃었을 때 몇 프레임까지 기다릴지 (약 0.5초 @30fps)
        self.declare_parameter('primary_lost_threshold_frames', 15)

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
        
        # [수정] 추적 안정성 파라미터 로드
        self.PRIMARY_LOST_THR = int(self.get_parameter('primary_lost_threshold_frames').value)

        # ---- State ----
        self.bridge = CvBridge()
        self.depth_frame = None
        self.fx = None; self.fy = None; self.cx_cam = None; self.cy_cam = None

        # [수정] 추적 상태를 기억하기 위한 변수들
        self.primary_id: Optional[int] = None # PointStamped로 publish 하는 고유 사용자 ID
        self.primary_lost_frames: int = 0     # primary_id를 잃은 후 지난 프레임 수
        self.last_goal: Optional[PointStamped] = None # 마지막 publish 값 (추정 유지용)

        # ---- Device / perf ----
        self.gpu = torch.cuda.is_available()
        self.device_str = 'cuda:0' if self.gpu else 'cpu'
        torch.backends.cudnn.benchmark = True

        # ---- YOLO load (person class only) ----
        # [수정] 안정적인 모델 저장/로드 경로 설정
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, ".cache/yolo_models")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, self.weight_name)

        if not os.path.exists(model_file):
            self.get_logger().warn(f"Downloading {self.weight_name} to {model_file}…")
            try:
                # [중요] 다운로드 URL은 공식 명칭을 사용해야 함
                urllib.request.urlretrieve(
                    f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
                    model_file
                )
            except Exception as e:
                self.get_logger().error(f"Failed to download YOLO model: {e}. Check model name ('{self.weight_name}')")
                rclpy.shutdown()
                return

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
            embedder='mobilenet',
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

        self.get_logger().info(f"✅ HumanTracker initialized (model: {self.weight_name}, conf: {self.conf_thr})")

    def depth_cb(self, msg: Image):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"[depth_cb] convert failed: {e}")

    def caminfo_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx_cam = msg.k[2]; self.cy_cam = msg.k[5]

    def _depth_from_bbox(self, bbox, depth_img) -> Optional[float]:
        x,y,w,h = map(int, bbox)
        H, W = depth_img.shape[:2]
        cx, cy = int(x + w/2), int(y + h/2)
        rw = max(8, int(w * self.depth_roi_frac))
        rh = max(8, int(h * self.depth_roi_frac))
        x0 = max(0, cx - rw//2); x1 = min(W, cx + rw//2)
        y0 = max(0, cy - rh//2); y1 = min(H, cy + rh//2)
        roi = depth_img[y0:y1, x0:x1]
        if roi.size == 0: return None
        valid = roi[roi > 0]
        if valid.size / float(roi.size) < self.depth_valid_ratio_min: return None
        scale = 0.001 if roi.dtype == np.uint16 else 1.0
        d = float(np.median(valid)) * scale
        return d if self.min_depth_m <= d <= self.max_depth_m else None

    def _project_to_camera(self, cx, cy, z_m):
        X = (cx - self.cx_cam) * z_m / self.fx
        Y = (cy - self.cy_cam) * z_m / self.fy
        return float(z_m), float(-X), float(-Y)

    # [수정] rgb_cb 함수 로직 전체를 안정적인 추적 로직으로 변경
    def rgb_cb(self, msg: Image):
        try:
            if self.depth_frame is None or self.fx is None:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            H, W = frame.shape[:2]

            # YOLO inference
            with torch.inference_mode():
                res = self.yolo.predict(
                    frame, classes=[0], imgsz=self.imgsz,
                    device=(0 if self.gpu else 'cpu'),
                    half=self.gpu, conf=self.conf_thr,
                    iou=self.iou_thr, verbose=False
                )[0]

            # Parse detections for DeepSORT
            dets_ds = []
            min_area = 0.001 * float(W * H)
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
                w, h = x2-x1, y2-y1
                if w*h > min_area and w > 1 and h > 1:
                    dets_ds.append(((x1, y1, w, h), float(b.conf[0]), None))

            # Tracking
            tracks = self.tracker.update_tracks(dets_ds, frame=frame)

            # Build current detections with depth
            current_detections: List[Detection] = []
            for tr in tracks:
                if not tr.is_confirmed(): continue
                x1,y1,x2,y2 = map(int, tr.to_tlbr())
                w, h = max(1, x2-x1), max(1, y2-y1)
                d = self._depth_from_bbox((x1, y1, w, h), self.depth_frame)
                if d is not None:
                    cx, cy = x1 + w/2.0, y1 + h/2.0
                    current_detections.append(Detection(tr.track_id, (x1,y1,w,h), (cx,cy), d))

            # --- Stateful Primary User Logic ---
            primary_detection: Optional[Detection] = None

            if self.primary_id is not None:
                # 1. 기존 primary_id를 찾아본다
                found = [d for d in current_detections if d.track_id == self.primary_id]
                if found:
                    primary_detection = found[0]
                    self.primary_lost_frames = 0 # 찾았으니 카운터 리셋
                else:
                    # 2. 못 찾았으면 lost 카운터 증가
                    self.primary_lost_frames += 1
                    if self.primary_lost_frames > self.PRIMARY_LOST_THR:
                        # 3. 임계값을 넘으면 primary_id를 포기하고 새로 찾을 준비
                        self.get_logger().warn(f"Primary ID {self.primary_id} lost. Reselecting...")
                        self.primary_id = None
                        self.primary_lost_frames = 0

            if self.primary_id is None and current_detections:
                # 4. primary_id가 없으면, 가장 가까운 사람을 새로 선택
                current_detections.sort(key=lambda z: z.depth_m)
                primary_detection = current_detections[0]
                self.primary_id = primary_detection.track_id
                self.get_logger().info(f"New primary ID selected: {self.primary_id}")

            # --- Publish & Draw ---
            if primary_detection:
                px, py, pz = self._project_to_camera(primary_detection.center[0], primary_detection.center[1], primary_detection.depth_m)
                pt = PointStamped()
                pt.header.stamp = msg.header.stamp
                pt.header.frame_id = 'camera_base'
                pt.point.x, pt.point.y, pt.point.z = px, py, pz
                self.user_pub.publish(pt)
                self.last_goal = pt

            if self.show_window:
                # 화면에 표시할 사람들을 거리순으로 정렬
                current_detections.sort(key=lambda z: z.depth_m)
                # 최대 2명까지만 표시
                for d in current_detections[:self.max_people]:
                    l, t, w, h = d.bbox
                    is_primary = (d.track_id == self.primary_id)
                    color = (0,0,255) if is_primary else (0,255,0)
                    cv2.rectangle(frame, (l,t), (l+w,t+h), color, 2)
                    label = ("USER " if is_primary else "ID ") + f"{d.track_id} {d.depth_m:.2f}m"
                    cv2.putText(frame, label, (l, max(20, t-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow('Human Tracker', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"[rgb_cb] exception: {e}", throttle_duration_sec=2.0)


    def destroy_node(self):
        try:
            if self.show_window:
                cv2.destroyAllWindows()
        except Exception: pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HumanTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()

