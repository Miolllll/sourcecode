#!/usr/bin/env python3
import math, heapq
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time

import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped

import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


def chaikin_smooth(points, iterations=1):
    if len(points) < 3 or iterations <= 0:
        return points
    pts = points[:]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = np.array(pts[i]), np.array(pts[i + 1])
            Q, R = 0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1
            new_pts.extend([tuple(Q), tuple(R)])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


class UserGoalPlanner(Node):
    def __init__(self):
        super().__init__('user_goal_planner')

        self._declare_and_load_params()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=10)
        self.map_sub  = self.create_subscription(OccupancyGrid, '/pointcloud_2d_map', self.map_cb, qos)
        self.goal_sub = self.create_subscription(PointStamped, '/user_tracking', self.goal_cb, qos)
        self.path_pub = self.create_publisher(Path, '/global_plan', qos)
        qos_map = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub_debug_map  = self.create_publisher(OccupancyGrid, '/planner/persistent_inflated_map', qos_map)
        
        self.current_goal: Optional[PointStamped] = None
        self.map_info: Optional[OccupancyGrid.info] = None
        self.map_frame_id: Optional[str] = None
        self.kernel = None
        self.obstacle_hits: Optional[np.ndarray] = None
        self.obstacle_age: Optional[np.ndarray] = None
        self.final_inflated_map: Optional[np.ndarray] = None

        # DT-based clearance cache for A*
        self._last_clearance_m: Optional[np.ndarray] = None
        
        self.timer = self.create_timer(self.tick_period, self.tick)
        self.get_logger().info("User Goal Planner (Persistent + Selective Inflation + Turn-Clearance Cost + Asym Front Padding) Initialized.")

    def _declare_and_load_params(self):
        # Persistence / masking / robot / inflation
        self.declare_parameter('hit_threshold', 2)
        self.declare_parameter('age_threshold_frames', 10)
        self.declare_parameter('user_mask_radius_m', 1.0)
        self.declare_parameter('robot_width_m', 0.90)
        self.declare_parameter('robot_length_m', 1.20)
        self.declare_parameter('inflation_mode', 'width')  # 'width' or 'circle'
        self.declare_parameter('inflation_margin_m', 0.10) # circle에서 사용
        self.declare_parameter('side_margin_m', 0.028)      # width에서 사용
        self.declare_parameter('gap_relax_cells', 1)
        self.declare_parameter('unknown_is_lethal', True)
        self.declare_parameter('occ_threshold', 50)

        # Kinematic start offset (rear axle as start)
        self.declare_parameter('use_footprint_offset', True)
        self.declare_parameter('footprint_offset_x_m', -0.87)
        self.declare_parameter('footprint_offset_y_m', 0.0)

        # Search / post-process
        self.declare_parameter('turn_penalty', 0.5)
        self.declare_parameter('shortcut_iterations', 2)
        self.declare_parameter('rdp_epsilon_m', 0.22)
        self.declare_parameter('smooth_iterations', 1)

        # Loop / bias
        self.declare_parameter('tick_period_s', 0.25)
        self.declare_parameter('use_center_bias', True)
        self.declare_parameter('center_bias_weight', 0.38)
        self.declare_parameter('center_bias_gamma', 2.0)

        # Selective inflation (small blobs won't inflate)
        self.declare_parameter('use_selective_inflation', True)
        self.declare_parameter('min_inflate_blob_area_m2', 0.015)

        # Turn clearance cost — width 모드 보완
        self.declare_parameter('use_turn_clearance_cost', True)
        self.declare_parameter('turn_clearance_margin_m', 0.05)
        self.declare_parameter('turn_clearance_gain', 2.0)
        self.declare_parameter('clearance_power', 2.0)

        # === Asymmetric front padding (NEW) ===
        self.declare_parameter('base_frame', 'rear_axle')
        self.declare_parameter('use_asym_front_padding', True)
        self.declare_parameter('front_left_extra_m',  0.15)   # 전방 좌측 여유
        self.declare_parameter('front_right_extra_m', 0.15)   # 전방 우측 여유(더 크게)
        self.declare_parameter('front_sector_deg',    75.0)   # 전방 부채꼴 각도(±)

        # Load
        self.hit_threshold = self.get_parameter('hit_threshold').value
        self.age_threshold = self.get_parameter('age_threshold_frames').value
        self.user_mask_radius = self.get_parameter('user_mask_radius_m').value
        self.robot_w = self.get_parameter('robot_width_m').value
        self.robot_l = self.get_parameter('robot_length_m').value
        self.inflation_mode = self.get_parameter('inflation_mode').value
        self.inflation_margin = self.get_parameter('inflation_margin_m').value
        self.side_margin = self.get_parameter('side_margin_m').value
        self.gap_relax_cells = self.get_parameter('gap_relax_cells').value
        self.unknown_lethal = self.get_parameter('unknown_is_lethal').value
        self.occ_th = self.get_parameter('occ_threshold').value
        self.use_footprint_offset = self.get_parameter('use_footprint_offset').value
        self.footprint_offset_x = self.get_parameter('footprint_offset_x_m').value
        self.footprint_offset_y = self.get_parameter('footprint_offset_y_m').value
        self.turn_pen = self.get_parameter('turn_penalty').value
        self.shortcut_iter = self.get_parameter('shortcut_iterations').value
        self.rdp_eps_m = self.get_parameter('rdp_epsilon_m').value
        self.smooth_iter = self.get_parameter('smooth_iterations').value
        self.tick_period = self.get_parameter('tick_period_s').value
        self.use_center_bias = self.get_parameter('use_center_bias').value
        self.center_bias_w = self.get_parameter('center_bias_weight').value
        self.center_bias_gamma = self.get_parameter('center_bias_gamma').value

        self.use_selective_inflation = bool(self.get_parameter('use_selective_inflation').value)
        self.min_inflate_blob_area_m2 = float(self.get_parameter('min_inflate_blob_area_m2').value)

        self.use_turn_clearance = bool(self.get_parameter('use_turn_clearance_cost').value)
        self.turn_clearance_margin = float(self.get_parameter('turn_clearance_margin_m').value)
        self.turn_clearance_gain = float(self.get_parameter('turn_clearance_gain').value)
        self.clearance_power = float(self.get_parameter('clearance_power').value)

        # Asym front padding load
        self.base_frame = self.get_parameter('base_frame').value
        self.use_asym_front = bool(self.get_parameter('use_asym_front_padding').value)
        self.front_left_extra  = float(self.get_parameter('front_left_extra_m').value)
        self.front_right_extra = float(self.get_parameter('front_right_extra_m').value)
        self.front_sector_deg  = float(self.get_parameter('front_sector_deg').value)

        self.circ_radius = 0.5 * math.sqrt(self.robot_w**2 + self.robot_l**2)

    def goal_cb(self, msg: PointStamped):
        self.current_goal = msg

    def map_cb(self, msg: OccupancyGrid):
        self.map_info = msg.info
        self.map_frame_id = msg.header.frame_id
        self._ensure_kernel(msg)
        self._initialize_persistence_maps_if_needed()

        new_occ_map = self._get_occ_from_msg(msg)
        masked_map = self._apply_user_mask(new_occ_map)
        persistent_map = self._update_persistence_maps(masked_map)

        # Selective inflation
        if self.use_selective_inflation:
            self.final_inflated_map = self._inflate_map_selective(persistent_map)
        else:
            self.final_inflated_map = self._inflate_map(persistent_map)

        self._publish_debug_map(self.final_inflated_map, self.map_info)

    def _initialize_persistence_maps_if_needed(self):
        if self.obstacle_hits is None and self.map_info:
            map_shape = (self.map_info.height, self.map_info.width)
            self.obstacle_hits = np.zeros(map_shape, dtype=np.int16)
            self.obstacle_age  = np.full(map_shape, self.age_threshold + 1, dtype=np.int16)

    def _get_occ_from_msg(self, msg: OccupancyGrid) -> np.ndarray:
        map_shape = (msg.info.height, msg.info.width)
        data = np.array(msg.data, dtype=np.int8).reshape(map_shape)
        occ_map = (data >= self.occ_th)
        if self.unknown_lethal:
            occ_map = np.logical_or(occ_map, data < 0)
        return occ_map

    def _apply_user_mask(self, occ_map: np.ndarray) -> np.ndarray:
        if self.current_goal is None:
            return occ_map
        try:
            i0, j0 = self.world_to_grid_idx(self.map_info, self.current_goal.point.x, self.current_goal.point.y)
            H, W = occ_map.shape
            radius_cells = int(self.user_mask_radius / self.map_info.resolution)
            y, x = np.ogrid[-j0:H-j0, -i0:W-i0]
            mask = x**2 + y**2 <= radius_cells**2
            masked_map = occ_map.copy()
            masked_map[mask] = False
            return masked_map
        except Exception:
            return occ_map

    def _update_persistence_maps(self, masked_map: np.ndarray) -> np.ndarray:
        self.obstacle_age += 1

        hit_mask = masked_map == True
        self.obstacle_hits[hit_mask] += 1
        self.obstacle_age[hit_mask] = 0

        miss_mask = masked_map == False
        self.obstacle_hits[miss_mask] = 0

        confirmed = self.obstacle_hits >= self.hit_threshold
        lingering = self.obstacle_age < self.age_threshold
        return np.logical_or(confirmed, lingering)

    def _inflate_map(self, blocked_map: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.kernel.size <= 1:
            return blocked_map
        return cv2.dilate(blocked_map.astype(np.uint8), self.kernel) > 0

    def _inflate_map_selective(self, blocked_map: np.ndarray) -> np.ndarray:
        """작은 블랍은 팽창하지 않고, 큰 블랍만 팽창. 최종 = 원본 OR 팽창(큰블랍)."""
        if self.kernel is None or self.kernel.size <= 1:
            return blocked_map

        num, labels = cv2.connectedComponents(blocked_map.astype(np.uint8))
        if num <= 1:
            return blocked_map  # no obstacles

        res = float(self.map_info.resolution)
        area_thr_cells = max(1, int(round(self.min_inflate_blob_area_m2 / (res * res))))

        counts = np.bincount(labels.flatten())
        big_ids = np.where(counts >= area_thr_cells)[0]
        big_ids = big_ids[big_ids != 0]  # exclude background

        if big_ids.size == 0:
            return blocked_map  # all blobs too small → no inflation

        big_mask = np.isin(labels, big_ids)
        inflated_big = cv2.dilate(big_mask.astype(np.uint8), self.kernel) > 0

        return np.logical_or(blocked_map, inflated_big)

    # ===================== Asymmetric Front Padding (helpers) =====================
    def _yaw_from_quat(self, q):
        # yaw-only 추출
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _apply_asym_front_padding(self, blocked_map: np.ndarray, start_ij: Tuple[int,int], yaw_rad: float, res: float) -> np.ndarray:
        """
        전방(±front_sector_deg) 부채꼴 안에서만 장애물 추가 팽창.
        좌/우 전방 섹터에 서로 다른 반경을 적용.
        """
        if not self.use_asym_front or (self.front_left_extra <= 1e-6 and self.front_right_extra <= 1e-6):
            return blocked_map

        H, W = blocked_map.shape
        i0, j0 = start_ij
        if not (0 <= i0 < W and 0 <= j0 < H):
            return blocked_map

        free_u8 = (~blocked_map).astype(np.uint8) * 255
        dist = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)  # pixels

        rL = max(0, int(round(self.front_left_extra  / max(res,1e-6))))
        rR = max(0, int(round(self.front_right_extra / max(res,1e-6))))
        if rL == 0 and rR == 0:
            return blocked_map

        jj, ii = np.indices((H, W)).astype(np.float32)
        dx = ii - float(i0)
        dy = jj - float(j0)
        ang = np.arctan2(dy, dx) - float(yaw_rad)
        ang = (ang + np.pi) % (2.0*np.pi) - np.pi

        sector = math.radians(float(self.front_sector_deg))
        front_mask = np.abs(ang) <= sector
        left_mask  = np.logical_and(front_mask, ang > 0.0)   # 전방-좌
        right_mask = np.logical_and(front_mask, ang <= 0.0)  # 전방-우

        extra_L = (dist <= rL) if rL > 0 else np.zeros_like(blocked_map, dtype=np.uint8)
        extra_R = (dist <= rR) if rR > 0 else np.zeros_like(blocked_map, dtype=np.uint8)

        add_L = np.logical_and(extra_L.astype(bool), left_mask)
        add_R = np.logical_and(extra_R.astype(bool), right_mask)

        return np.logical_or(blocked_map, np.logical_or(add_L, add_R))
    # ==============================================================================

    def tick(self):
        self.try_plan()
        
    def try_plan(self):
        if self.final_inflated_map is None or self.current_goal is None or self.map_info is None or not self.map_frame_id:
            return

        final_blocked_map = self.final_inflated_map.copy()
        grid_info = self.map_info
        frame = self.map_frame_id
        origin_x, origin_y, res = grid_info.origin.position.x, grid_info.origin.position.y, grid_info.resolution
        
        # Start pose (rear axle or camera)
        if self.use_footprint_offset:
            start_x, start_y = self.footprint_offset_x, self.footprint_offset_y
        else:
            start_x, start_y = 0.0, 0.0

        # Transform goal into map frame
        try:
            trans = self.tf_buffer.lookup_transform(frame, self.current_goal.header.frame_id, Time())
            goal_in_map = do_transform_point(self.current_goal, trans)
            goal_x, goal_y = float(goal_in_map.point.x), float(goal_in_map.point.y)
        except Exception as e:
            self.get_logger().warn(f"[TF] goal transform failed: {e}")
            return

        # === Asymmetric front padding 적용 (start 기준, yaw 사용) ===
        try:
            tf_bl = self.tf_buffer.lookup_transform(self.map_frame_id, self.base_frame, Time())
            yaw = self._yaw_from_quat(tf_bl.transform.rotation)
        except Exception:
            yaw = 0.0  # TF 실패 시 가정

        start_idx_for_clear = self.world_to_grid_idx(grid_info, start_x, start_y)
        final_blocked_map = self._apply_asym_front_padding(final_blocked_map, start_idx_for_clear, yaw, res)

        # Clear start footprint (so we don't think we are colliding with ourselves)
        i0, j0 = start_idx_for_clear
        robot_w_cells = int(self.robot_w / res)
        robot_l_cells = int(self.robot_l / res)
        half_w = robot_w_cells // 2
        min_j, max_j = j0, j0 + robot_l_cells
        min_i, max_i = i0 - half_w, i0 + half_w
        H, W = final_blocked_map.shape
        min_j_c, max_j_c = max(0, min_j), min(H, max_j)
        min_i_c, max_i_c = max(0, min_i), min(W, max_i)
        final_blocked_map[min_j_c:max_j_c, min_i_c:max_i_c] = False
        
        start_idx = self.world_to_grid_idx(grid_info, start_x, start_y)
        goal_idx  = self.world_to_grid_idx(grid_info, goal_x, goal_y)
        start_idx = self._snap_if_blocked(final_blocked_map, start_idx)
        goal_idx  = self._snap_if_blocked(final_blocked_map, goal_idx)
        if start_idx is None:
            self.get_logger().warn("[PLAN] start index is blocked and cannot be snapped to free.")
            return
        if goal_idx is None:
            self.get_logger().warn("[PLAN] goal index is blocked and cannot be snapped to free.")
            return

        # ----- Clearance & center-bias costs -----
        extra_cost = None
        self._last_clearance_m = None
        if self.use_center_bias or self.use_turn_clearance:
            free = (~final_blocked_map).astype(np.uint8) * 255
            dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)  # pixels
            res_m = float(res)
            self._last_clearance_m = dist * res_m               # meters
            if self.use_center_bias:
                m = float(dist.max()) if dist.size > 0 else 1.0
                if m < 1e-6: m = 1.0
                dist_n = dist / m
                extra_cost = (self.center_bias_w * np.power(1.0 - dist_n, self.center_bias_gamma)).astype(np.float32)

        # ----- A* plan -----
        path_idx = self.astar(final_blocked_map, start_idx, goal_idx, extra_cost)
        if not path_idx:
            self.get_logger().warn("[PLAN] A* failed to find a path.")
            return

        # Post-process
        path_idx = self._shortcut(path_idx, final_blocked_map, iters=self.shortcut_iter)
        world_pts = [(origin_x + (i + 0.5) * res, origin_y + (j + 0.5) * res) for (i, j) in path_idx]
        if self.rdp_eps_m > 1e-6:
            world_pts = self._rdp(world_pts, self.rdp_eps_m)
        if self.smooth_iter > 0:
            world_pts = chaikin_smooth(world_pts, self.smooth_iter)

        # Publish
        plan = Path()
        plan.header.frame_id = frame
        plan.header.stamp    = self.get_clock().now().to_msg()
        plan.poses = [self._create_pose(x, y, plan.header) for x, y in world_pts]
        self.path_pub.publish(plan)
    
    def _create_pose(self, x, y, header):
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        return pose

    def _ensure_kernel(self, msg: OccupancyGrid):
        res = float(msg.info.resolution)
        if self.kernel is not None and hasattr(self, 'last_res') and abs(self.last_res - res) < 1e-9:
            return
        self.last_res = res
        if self.inflation_mode == 'width':
            rad_m = 0.5 * self.robot_w + self.side_margin
        else:
            rad_m = self.circ_radius + self.inflation_margin
        cells = int(math.ceil(rad_m / max(res, 1e-6))) - self.gap_relax_cells
        size = 2 * max(0, cells) + 1
        if size < 1:
            size = 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _snap_if_blocked(self, blocked: np.ndarray, idx: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        if idx is None:
            return None
        i, j = idx
        H, W = blocked.shape
        if not (0 <= i < W and 0 <= j < H):
            return None
        if not blocked[j, i]:
            return (i, j)
        from collections import deque
        nbrs8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        q = deque([(i, j)])
        seen = set([(i, j)])
        while q:
            ci, cj = q.popleft()
            for di, dj in nbrs8:
                ni, nj = ci+di, cj+dj
                if 0 <= ni < W and 0 <= nj < H and (ni, nj) not in seen:
                    if not blocked[nj, ni]:
                        return (ni, nj)
                    seen.add((ni, nj))
                    q.append((ni, nj))
        return None

    def astar(self, blocked: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], extra_cost: Optional[np.ndarray]):
        H, W = blocked.shape
        if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
            return []
        nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        step_cost = [1.0,1.0,1.0,1.0,math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]
        def h(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

        # 회전시 필요한 최소 여유(미터): 로봇 반폭 + 여유
        req_clear_m = 0.5 * float(self.robot_w) + float(self.turn_clearance_margin)

        openq = [(h(start,goal), 0.0, start, None)]
        gscore = {start: 0.0}
        came = {}
        while openq:
            f,g,cur,parent = heapq.heappop(openq)
            if cur in came:
                continue
            came[cur] = parent
            if cur == goal:
                path = []
                n = cur
                while n is not None:
                    path.append(n)
                    n = came.get(n)
                return path[::-1]
            for k,(di,dj) in enumerate(nbrs):
                n = (cur[0]+di, cur[1]+dj)
                if not (0 <= n[0] < W and 0 <= n[1] < H) or blocked[n[1], n[0]]:
                    continue
                ng = g + step_cost[k]
                if extra_cost is not None:
                    ng += float(extra_cost[n[1], n[0]])

                # 턴 여부
                turning = (parent is not None) and ((n[0]-cur[0] != cur[0]-parent[0]) or (n[1]-cur[1] != cur[1]-parent[1]))
                if turning:
                    ng += float(self.turn_pen)
                    # 회전-클리어런스 패널티
                    if self.use_turn_clearance and (self._last_clearance_m is not None):
                        cfree = float(self._last_clearance_m[n[1], n[0]])  # 이 칸에서 장애물까지 거리(미터)
                        shortfall = max(0.0, req_clear_m - cfree)
                        if shortfall > 0.0:
                            ng += float(self.turn_clearance_gain) * (shortfall ** float(self.clearance_power))

                if ng < gscore.get(n, float('inf')):
                    gscore[n] = ng
                    heapq.heappush(openq, (ng + h(n,goal), ng, n, cur))
        return []

    def world_to_grid_idx(self, grid_info, x: float, y: float) -> Tuple[int, int]:
        return (int((x - grid_info.origin.position.x) / grid_info.resolution),
                int((y - grid_info.origin.position.y) / grid_info.resolution))

    def _los_free(self, a: Tuple[int,int], b: Tuple[int,int], blocked: np.ndarray) -> bool:
        (i0,j0),(i1,j1) = a,b
        di, dj = abs(i1-i0), -abs(j1-j0)
        si = 1 if i0<i1 else -1
        sj = 1 if j0<j1 else -1
        err = di+dj
        while True:
            if blocked[j0,i0]:
                return False
            if i0 == i1 and j0 == j1:
                break
            e2 = 2*err
            if e2 >= dj:
                err += dj
                i0 += si
            if e2 <= di:
                err += di
                j0 += sj
        return True

    def _shortcut(self, path_idx: List[Tuple[int,int]], blocked: np.ndarray, iters: int = 2):
        pts = path_idx
        for _ in range(iters):
            if len(pts) < 3:
                break
            new_pts = [pts[0]]
            last_pt_idx = 0
            while last_pt_idx < len(pts) - 1:
                best_next_idx = last_pt_idx + 1
                for next_idx in range(len(pts) - 1, last_pt_idx + 1, -1):
                    if self._los_free(pts[last_pt_idx], pts[next_idx], blocked):
                        best_next_idx = next_idx
                        break
                new_pts.append(pts[best_next_idx])
                last_pt_idx = best_next_idx
            pts = new_pts
        return pts
        
    def _rdp(self, points: List[Tuple[float,float]], eps: float) -> List[Tuple[float,float]]:
        if len(points) < 3:
            return points
        dmax, index = 0.0, 0
        end = len(points) - 1
        for i in range(1, end):
            d = self._point_line_distance(points[i], points[0], points[end])
            if d > dmax:
                index, dmax = i, d
        if dmax > eps:
            left  = self._rdp(points[:index+1], eps)
            right = self._rdp(points[index:], eps)
            return left[:-1] + right
        else:
            return [points[0], points[end]]

    def _point_line_distance(self, p, p1, p2):
        x0, y0 = p
        x1, y1 = p1
        x2, y2 = p2
        return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / math.hypot(y2-y1, x2-x1)

    def _publish_debug_map(self, blocked: np.ndarray, grid_info: OccupancyGrid.info):
        if grid_info is None or blocked is None or not self.map_frame_id:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.info = grid_info
        out = np.zeros_like(blocked, dtype=np.int8)
        out[blocked] = 100
        msg.data = out.flatten().tolist()
        self.pub_debug_map.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UserGoalPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()