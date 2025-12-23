#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs               # PointStamped 변환 지원
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
import heapq

class UserGoalPlanner(Node):
    def __init__(self):
        super().__init__('user_goal_planner')

        # TF buffer/listener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscriptions & publisher
        self.map_sub  = self.create_subscription(OccupancyGrid, '/pointcloud_2d_map', self.map_cb,  10)
        self.goal_sub = self.create_subscription(PointStamped,   '/user_tracking',    self.goal_cb, 10)
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)

        self.latest_map   = None
        self.current_goal = None

    def map_cb(self, msg: OccupancyGrid):
        self.get_logger().info('map_cb: OccupancyGrid received')
        self.latest_map = msg
        self.try_plan()

    def goal_cb(self, msg: PointStamped):
        pt = msg.point
        self.get_logger().info(
            f"goal_cb: PointStamped in '{msg.header.frame_id}' -> "
            f"({pt.x:.2f},{pt.y:.2f},{pt.z:.2f})"
        )
        self.current_goal = msg
        self.try_plan()

    def try_plan(self):
        if self.latest_map is None or self.current_goal is None:
            return

        self.get_logger().info('try_plan: starting')
        grid = self.latest_map
        frame = grid.header.frame_id  # 'camera_base'
        w, h = grid.info.width, grid.info.height
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        res = grid.info.resolution

        # 1) 시작점: camera_base 프레임 원점 고정
        start_x, start_y = 0.0, 0.0

        # 2) 목표점 → camera_base 로 변환
        try:
            trans = self.tf_buffer.lookup_transform(
                frame,
                self.current_goal.header.frame_id,
                Time()
            )
            goal_in_cam = do_transform_point(self.current_goal, trans)
            goal_x = goal_in_cam.point.x
            goal_y = goal_in_cam.point.y
        except Exception as e:
            self.get_logger().warn(f"[TF] goal transform failed: {e}")
            return

        # 3) 월드→그리드 인덱스
        start_idx = self.world_to_grid(grid, start_x, start_y)
        goal_idx  = self.world_to_grid(grid, goal_x, goal_y)

        # 4) 범위 검사
        if not (0<=start_idx[0]<w and 0<=start_idx[1]<h): 
            self.get_logger().warn(f"start_idx {start_idx} OOB"); return
        if not (0<=goal_idx[0]<w and 0<=goal_idx[1]<h):   
            self.get_logger().warn(f"goal_idx  {goal_idx} OOB"); return

        # 5) A* 탐색
        raw_path = self.astar(grid, start_idx, goal_idx)
        if not raw_path:
            self.get_logger().warn('A* returned no path'); return

        # 6) 그리드 인덱스 → 월드 좌표 리스트
        world_pts = [
            (origin_x + (i+0.5)*res, origin_y + (j+0.5)*res)
            for i,j in raw_path
        ]

        # 7) Chaikin smoothing (반복 횟수 n)
        smooth_pts = self.chaikin_smoothing(world_pts, iterations=3)

        # 8) Path 메시지 생성
        plan = Path()
        plan.header.frame_id = frame
        plan.header.stamp    = self.get_clock().now().to_msg()
        for x,y in smooth_pts:
            pose = PoseStamped()
            pose.header = plan.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            plan.poses.append(pose)

        self.path_pub.publish(plan)
        self.get_logger().info(f"Published smoothed path with {len(plan.poses)} points")

    def world_to_grid(self, grid: OccupancyGrid, x: float, y: float):
        i = int((x - grid.info.origin.position.x) / grid.info.resolution)
        j = int((y - grid.info.origin.position.y) / grid.info.resolution)
        return (i, j)

    def astar(self, grid: OccupancyGrid, start, goal):
        w, h = grid.info.width, grid.info.height
        data = grid.data
        # 8-방향 이웃
        nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        def h_cost(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        open_set = [(h_cost(start,goal), 0, start, None)]
        came_from = {}
        g_score = {start:0}
        visited = set()

        while open_set:
            f, g, cur, parent = heapq.heappop(open_set)
            if cur in visited: continue
            visited.add(cur)
            came_from[cur] = parent
            if cur == goal: break

            for dx,dy in nbrs:
                nbr = (cur[0]+dx, cur[1]+dy)
                if not (0<=nbr[0]<w and 0<=nbr[1]<h): continue
                occ = data[nbr[1]*w + nbr[0]]
                if occ == 100: continue
                cost = g + (1.4 if dx and dy else 1.0)
                if cost < g_score.get(nbr, float('inf')):
                    g_score[nbr] = cost
                    heapq.heappush(open_set, (cost+h_cost(nbr,goal), cost, nbr, cur))

        if goal not in came_from: return []
        path = []
        node = goal
        while node:
            path.append(node)
            node = came_from.get(node)
        return path[::-1]

    def chaikin_smoothing(self, pts, iterations=2):
        """
        Chaikin's corner cutting algorithm
        pts: [(x,y), ...]
        """
        for _ in range(iterations):
            new_pts = []
            for i in range(len(pts)-1):
                p0, p1 = pts[i], pts[i+1]
                Q = (0.75*p0[0]+0.25*p1[0], 0.75*p0[1]+0.25*p1[1])
                R = (0.25*p0[0]+0.75*p1[0], 0.25*p0[1]+0.75*p1[1])
                new_pts += [Q, R]
            pts = new_pts
        return pts

def main(args=None):
    rclpy.init(args=args)
    node = UserGoalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
