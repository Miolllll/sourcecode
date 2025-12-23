#!/usr/bin/env python3
import math, time, heapq
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped

import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_point


# ----------------- utilities -----------------
def chaikin_smooth(points: List[Tuple[float, float]], iterations: int = 1) -> List[Tuple[float, float]]:
    if len(points) < 3 or iterations <= 0:
        return points
    pts = points[:]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = np.array(pts[i]); p1 = np.array(pts[i + 1])
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_pts.extend([tuple(Q), tuple(R)])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def yaw_from_quat(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class UserGoalPlannerSafe(Node):
    def __init__(self):
        super().__init__("user_goal_planner_safe")

        # -------- Parameters --------
        self.declare_parameter("map_topic",   "/pointcloud_2d_map")
        self.declare_parameter("goal_topic",  "/user_tracking")     # PointStamped
        self.declare_parameter("path_topic",  "/global_plan")
        self.declare_parameter("map_frame",   "map")
        self.declare_parameter("base_frame",  "base_link")          # camera_base도 가능

        # robot footprint
        self.declare_parameter("robot_width_m",   0.90)
        self.declare_parameter("robot_length_m",  1.20)
        self.declare_parameter("inflation_margin_m", 0.15)

        # memory / unknown
        self.declare_parameter("obstacle_ttl_s", 3.0)
        self.declare_parameter("unknown_is_lethal", True)

        # smoothing(출력 경로용)
        self.declare_parameter("smooth_iterations", 1)

        # debug map
        self.declare_parameter("publish_debug_map", True)

        # camera_base → footprint 중심 보정(옵션)
        self.declare_parameter("use_footprint_offset", False)
        self.declare_parameter("footprint_offset_x_m", 0.0)
        self.declare_parameter("footprint_offset_y_m", 0.0)

        # 곡선 품질 옵션(그대로 유지)
        self.declare_parameter("use_center_bias", True)
        self.declare_parameter("center_bias_weight", 0.6)
        self.declare_parameter("turn_penalty", 0.3)
        self.declare_parameter("shortcut_iterations", 2)
        self.declare_parameter("rdp_epsilon_m", 0.15)

        # ★ 작은 군집 제거(occ/unknown 각각)
        self.declare_parameter("min_cluster_cells", 16)              # occ용(이전)
        self.declare_parameter("min_unknown_cluster_cells", 2)      # unknown용(신규)
        # ★ unknown salt-noise 제거용 오프닝 커널(반경=셀, 0=끄기)
        self.declare_parameter("unknown_opening_kernel_cells", 1.0)

        # -------- Load --------
        self.map_topic   = self.get_parameter("map_topic").value
        self.goal_topic  = self.get_parameter("goal_topic").value
        self.path_topic  = self.get_parameter("path_topic").value
        self.map_frame   = self.get_parameter("map_frame").value
        self.base_frame  = self.get_parameter("base_frame").value

        self.robot_w  = float(self.get_parameter("robot_width_m").value)
        self.robot_l  = float(self.get_parameter("robot_length_m").value)
        self.margin   = float(self.get_parameter("inflation_margin_m").value)
        self.ttl_s    = float(self.get_parameter("obstacle_ttl_s").value)
        self.unknown_lethal = bool(self.get_parameter("unknown_is_lethal").value)
        self.smooth_iter    = int(self.get_parameter("smooth_iterations").value)
        self.publish_debug  = bool(self.get_parameter("publish_debug_map").value)

        self.use_offset = bool(self.get_parameter("use_footprint_offset").value)
        self.off_x = float(self.get_parameter("footprint_offset_x_m").value)
        self.off_y = float(self.get_parameter("footprint_offset_y_m").value)

        self.use_center_bias = bool(self.get_parameter("use_center_bias").value)
        self.center_bias_w   = float(self.get_parameter("center_bias_weight").value)
        self.turn_pen        = float(self.get_parameter("turn_penalty").value)
        self.shortcut_iter   = int(self.get_parameter("shortcut_iterations").value)
        self.rdp_eps_m       = float(self.get_parameter("rdp_epsilon_m").value)

        self.min_cluster_cells_occ     = int(self.get_parameter("min_cluster_cells").value)
        self.min_cluster_cells_unknown = int(self.get_parameter("min_unknown_cluster_cells").value)
        self.unknown_opening_k         = int(self.get_parameter("unknown_opening_kernel_cells").value)

        # 팽창 반경(외접반경 + 여유)
        self.circ_radius = 0.5 * math.sqrt(self.robot_w**2 + self.robot_l**2)

        # -------- TF --------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Pub/Sub --------
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, qos)
        self.goal_sub = self.create_subscription(PointStamped,   self.goal_topic, self.goal_cb, qos)
        self.pub_path = self.create_publisher(Path, self.path_topic, qos)
        self.pub_debug_map = self.create_publisher(OccupancyGrid, "/planner/ttl_inflated_map", qos)

        # -------- Persistent grid state --------
        self.map_msg: Optional[OccupancyGrid] = None
        self.res: Optional[float] = None
        self.origin_x = self.origin_y = 0.0
        self.W = self.H = 0

        self.occ_mask: Optional[np.ndarray] = None
        self.free_mask: Optional[np.ndarray] = None
        self.last_seen: Optional[np.ndarray] = None

        self.inflation_cells: Optional[int] = None
        self.kernel: Optional[np.ndarray] = None

        self.last_goal: Optional[PointStamped] = None
        self.timer = self.create_timer(0.2, self.tick)  # 5 Hz

        self.get_logger().info(
            f"[planner_safe] base_frame={self.base_frame}, TTL={self.ttl_s}s, "
            f"unknown_lethal={self.unknown_lethal}, robot(circ)={self.circ_radius:.2f}m + margin={self.margin:.2f}m, "
            f"use_offset={self.use_offset}, off=({self.off_x:.3f},{self.off_y:.3f}), "
            f"center_bias={self.use_center_bias}:{self.center_bias_w}, turn_pen={self.turn_pen}, "
            f"min_cluster_cells_occ={self.min_cluster_cells_occ}, "
            f"min_cluster_cells_unknown={self.min_cluster_cells_unknown}, "
            f"unknown_opening_k={self.unknown_opening_k}"
        )

    # ---------- helpers ----------
    def _init_grids(self, msg: OccupancyGrid):
        self.res = float(msg.info.resolution)
        self.W   = int(msg.info.width)
        self.H   = int(msg.info.height)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)

        self.occ_mask  = np.zeros((self.H, self.W), dtype=bool)
        self.free_mask = np.zeros((self.H, self.W), dtype=bool)
        self.last_seen = np.zeros((self.H, self.W), dtype=np.float32)

        total_inflation = self.circ_radius + self.margin
        cells = max(1, int(math.ceil(total_inflation / self.res)))
        self.inflation_cells = cells
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*cells+1, 2*cells+1))

        self.get_logger().info(
            f"[planner_safe] map {self.W}x{self.H}, res={self.res:.3f} m/cell, inflation={cells} cells"
        )

    def _msg_to_grid(self, msg: OccupancyGrid) -> np.ndarray:
        return np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

    def _world_to_rc(self, x: float, y: float) -> Tuple[int, int]:
        c = int((x - self.origin_x) / self.res)
        r = int((y - self.origin_y) / self.res)
        return r, c

    def _rc_to_world(self, r: int, c: int) -> Tuple[float, float]:
        x = self.origin_x + (c + 0.5) * self.res
        y = self.origin_y + (r + 0.5) * self.res
        return x, y

    # ---------- subscriptions ----------
    def map_cb(self, msg: OccupancyGrid):
        reinit = (
            self.map_msg is None or self.res is None or
            msg.info.width != self.W or msg.info.height != self.H or
            abs(msg.info.resolution - (self.res or 0.0)) > 1e-9 or
            abs(msg.info.origin.position.x - self.origin_x) > (self.res or 0.0) or
            abs(msg.info.origin.position.y - self.origin_y) > (self.res or 0.0)
        )
        if reinit:
            self._init_grids(msg)

        grid = self._msg_to_grid(msg)
        now = time.time()

        occ_new  = grid >= 50
        free_new = grid == 0

        self.free_mask[:] = free_new

        self.occ_mask[occ_new] = True
        self.last_seen[occ_new] = now

        clear_mask = np.logical_and(self.occ_mask, free_new)
        self.occ_mask[clear_mask] = False
        self.last_seen[clear_mask] = 0.0

        stale_mask = np.logical_and(self.occ_mask, np.logical_not(np.logical_or(occ_new, free_new)))
        if np.any(stale_mask):
            expire = (now - self.last_seen) > self.ttl_s
            to_clear = np.logical_and(stale_mask, expire)
            self.occ_mask[to_clear] = False
            self.last_seen[to_clear] = 0.0

        self.map_msg = msg

    def goal_cb(self, msg: PointStamped):
        self.last_goal = msg

    # ---------- small island removal ----------
    def _remove_small_islands(self, mask_bool: np.ndarray, min_cells: int) -> np.ndarray:
        if min_cells <= 1:
            return mask_bool
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
        if num <= 1:
            return mask_bool
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_ids = np.where(areas >= min_cells)[0] + 1
        return np.isin(labels, keep_ids)

    # ---------- planning ----------
    def _build_blocked_mask(self) -> Optional[np.ndarray]:
        if self.map_msg is None or self.kernel is None:
            return None

        # 1) 점유(occ)에서 작은 군집 제거
        occ = self.occ_mask.copy()
        if self.min_cluster_cells_occ > 0:
            occ = self._remove_small_islands(occ, self.min_cluster_cells_occ)

        # 2) unknown 처리
        if self.unknown_lethal:
            raw = self._msg_to_grid(self.map_msg)
            unknown = (raw < 0)

            # 2-1) 작은 unknown 군집 제거
            if self.min_cluster_cells_unknown > 0:
                unknown = self._remove_small_islands(unknown, self.min_cluster_cells_unknown)

            # 2-2) 모폴로지 오프닝(소금·후추 잡음 제거)
            if self.unknown_opening_k > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2*self.unknown_opening_k+1, 2*self.unknown_opening_k+1))
                unknown = cv2.morphologyEx(unknown.astype(np.uint8), cv2.MORPH_OPEN, k) > 0

            occ = np.logical_or(occ, unknown)

        # 3) 팽창 → blocked
        occ_uint8 = occ.astype(np.uint8) * 255
        inflated = cv2.dilate(occ_uint8, self.kernel)
        blocked = inflated > 0
        return blocked

    def _astar(self, start_rc, goal_rc, blocked, extra_cost=None, turn_penalty: float = 0.0):
        H, W = blocked.shape
        def inb(r,c): return (0 <= r < H and 0 <= c < W)
        if not inb(*start_rc) or not inb(*goal_rc): return None
        if blocked[start_rc] or blocked[goal_rc]:   return None

        dirs  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        costs = [1.0,   1.0,  1.0,   1.0,   math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
        def h(rc):
            dr = rc[0]-goal_rc[0]; dc = rc[1]-goal_rc[1]
            return math.hypot(dr, dc)

        g = {}; f = {}; came = {}; came_dir = {}
        start = start_rc
        g[start] = 0.0; f[start] = h(start)
        openq = [(f[start], start)]
        closed = set()

        while openq:
            _, cur = heapq.heappop(openq)
            if cur in closed: continue
            if cur == goal_rc:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            closed.add(cur)

            r,c = cur
            for k,(dr,dc) in enumerate(dirs):
                nr, nc = r+dr, c+dc
                n = (nr, nc)
                if not inb(nr,nc) or blocked[n]: continue

                step = costs[k]
                add  = 0.0
                if extra_cost is not None:
                    add += float(extra_cost[n])
                if cur in came_dir and came_dir[cur] != k:
                    add += float(turn_penalty)

                tentative = g[cur] + step + add
                if tentative < g.get(n, 1e18):
                    g[n] = tentative
                    f[n] = tentative + h(n)
                    came[n] = cur
                    came_dir[n] = k
                    heapq.heappush(openq, (f[n], n))
        return None

    def _tf_goal_to_map(self, goal: PointStamped) -> PointStamped:
        try:
            if goal.header.frame_id == self.map_frame:
                return goal
            tf = self.tf_buffer.lookup_transform(self.map_frame, goal.header.frame_id, rclpy.time.Time())
            return do_transform_point(goal, tf)
        except Exception as e:
            self.get_logger().warn(f"goal transform failed, using raw coords: {e}")
            return goal

    def _get_robot_pose_in_map(self) -> Optional[Tuple[float,float]]:
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

            if self.use_offset and (abs(self.off_x) > 1e-6 or abs(self.off_y) > 1e-6):
                cos_y = math.cos(yaw); sin_y = math.sin(yaw)
                dx = cos_y * self.off_x - sin_y * self.off_y
                dy = sin_y * self.off_x + cos_y * self.off_y
                x += dx; y += dy
            return (x, y)
        except Exception as e:
            self.get_logger().warn(f"robot pose transform failed: {e}")
            return None

    def _publish_debug_map(self, blocked: np.ndarray):
        if not self.publish_debug:
            return
        dbg = OccupancyGrid()
        dbg.header.stamp = self.get_clock().now().to_msg()
        dbg.header.frame_id = self.map_frame
        dbg.info = self.map_msg.info
        grid = np.full((self.H, self.W), 0, dtype=np.int16)
        grid[blocked] = 100
        dbg.data = grid.flatten().tolist()
        self.pub_debug_map.publish(dbg)

    # --- LOS / RDP / Shortcut (그대로) ---
    def _los_free_rc(self, a: Tuple[int,int], b: Tuple[int,int], blocked: np.ndarray) -> bool:
        (r0,c0),(r1,c1) = a,b
        dr = abs(r1-r0); dc = abs(c1-c0)
        sr = 1 if r1>=r0 else -1
        sc = 1 if c1>=c0 else -1
        err = dr - dc
        r,c = r0,c0
        H,W = blocked.shape
        while True:
            if not (0<=r<H and 0<=c<W) or blocked[r,c]:
                return False
            if (r,c)==(r1,c1): break
            e2 = 2*err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc
        return True

    def _shortcut(self, path_idx: List[Tuple[int,int]], blocked: np.ndarray, iters: int = 2):
        pts = list(path_idx)
        for _ in range(max(0, iters)):
            out=[pts[0]]; i=0
            while i < len(pts)-1:
                j = i+1
                while j < len(pts) and self._los_free_rc(pts[i], pts[j], blocked):
                    j += 1
                out.append(pts[j-1]); i = j-1
            pts = out
        return pts

    def _rdp(self, points: List[Tuple[float,float]], eps: float) -> List[Tuple[float,float]]:
        if len(points) < 3: return points
        (x1,y1),(x2,y2) = points[0], points[-1]
        vx,vy = x2-x1, y2-y1
        vlen = math.hypot(vx,vy) or 1e-9
        max_d, idx = -1.0, -1
        for i in range(1, len(points)-1):
            px,py = points[i][0]-x1, points[i][1]-y1
            d = abs(vx*py - vy*px)/vlen
            if d > max_d: max_d, idx = d, i
        if max_d > eps:
            left  = self._rdp(points[:idx+1], eps)
            right = self._rdp(points[idx:], eps)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]

    def tick(self):
        if self.map_msg is None or self.kernel is None:
            return
        blocked = self._build_blocked_mask()
        if blocked is None:
            return

        self._publish_debug_map(blocked)

        if self.last_goal is None:
            return

        goal_map = self._tf_goal_to_map(self.last_goal)
        start_xy = self._get_robot_pose_in_map()
        if start_xy is None:
            return

        sx, sy = start_xy
        gx, gy = goal_map.point.x, goal_map.point.y

        sr, sc = self._world_to_rc(sx, sy)
        gr, gc = self._world_to_rc(gx, gy)

        extra_cost = None
        if self.use_center_bias:
            free_mask = (~blocked).astype(np.uint8) * 255
            dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
            m = float(dist.max()) if dist.size else 1.0
            if m < 1e-6: m = 1.0
            extra_cost = self.center_bias_w * (1.0 - (dist / m))

        path_idx = self._astar((sr,sc), (gr,gc), blocked,
                               extra_cost=extra_cost,
                               turn_penalty=self.turn_pen)
        if path_idx is None or len(path_idx) < 2:
            return

        path_idx = self._shortcut(path_idx, blocked, iters=self.shortcut_iter)

        pts = [self._rc_to_world(r,c) for (r,c) in path_idx]

        if self.rdp_eps_m > 1e-6 and len(pts) > 2:
            pts = self._rdp(pts, self.rdp_eps_m)

        if self.smooth_iter > 0:
            pts = chaikin_smooth(pts, min(self.smooth_iter, 1))

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        for (x,y) in pts:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UserGoalPlannerSafe()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
