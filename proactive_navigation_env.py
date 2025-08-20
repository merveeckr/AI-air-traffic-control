"""
Proaktif Aircraft Navigation Environment
Waypoint tahmini + proaktif risk kaçınma
Güncellenmiş sürüm
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from typing import Tuple, Dict, Any, Optional

class ProactiveNavigationEnv(gym.Env):
    """Proaktif navigasyon ortamı - Waypoint tahmini + manevra planlama"""
    
    def __init__(self, prediction_horizon=3.0, dt=0.1, max_steps=1000, render_mode=None):  # Daha fazla adım
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.max_steps = max_steps
        self.max_turn_rate = np.deg2rad(45)
        self.max_speed = 20.0
        self.min_speed = 10.0
        self.safe_margin = 35.0
        self.goal_radius = 15.0  # Daha küçük hedef yarıçapı - daha yakına gelmeli
        self.world_bounds = {'x_min': -100, 'x_max': 200, 'y_min': -100, 'y_max': 200}
        # Curriculum + dynamic difficulty
        self.curriculum_num_clusters = 0
        self.success_count = 0
        self.dynamic_difficulty = True
        # Observation mode: 'v1' => 30-dim (backward compatible), 'v2' => 36-dim (extra cluster-relative features)
        self.obs_mode = 'v1'
        # Success policy knobs
        self.allow_los_success = False  # False: Erken başarıyı devre dışı bırak - daha yakına gelmeli
        self.allow_timeout_near_goal_success = True  # if True, allow success on time-limit when near goal
        # Training knobs (set by trainer when needed)
        self.training_mode = False
        self.initial_random_steps = 0
        self.observation_noise_std = 0.0

        # Backward-compatible observation shape (30)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.time = 0.0
        self.x, self.y, self.speed, self.heading = 0.0, 0.0, 15.0, 0.0
        # Dynamic difficulty knobs
        if self.dynamic_difficulty:
            self.cluster_radius = float(np.random.uniform(4.0, 7.0))
            base_clusters = int(np.random.randint(4, 8))
            # Curriculum escalation based on accumulated successes
            if self.success_count > 300:
                base_clusters = max(base_clusters, 7)
            elif self.success_count > 100:
                base_clusters = max(base_clusters, 5)
            self.curriculum_num_clusters = base_clusters
        self.clusters = self._sample_clusters()
        # Distant goal (encourage longer planning)
        if self.dynamic_difficulty:
            gx = float(np.random.uniform(60.0, 80.0))
            gy = float(np.random.uniform(60.0, 80.0))
            self.goal = np.array([gx, gy], dtype=np.float32)
        else:
            self.goal = self._sample_goal()
        self.episode_rewards = []
        self.episode_positions = [(self.x, self.y)]
        self.consecutive_risk_steps = 0
        self.consecutive_maneuver_steps = 0
        self.consecutive_safe_steps = 0
        # Stuck detection
        self.best_goal_dist = self._goal_dist()
        self.last_improve_step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Early random actions for exploration when training
        if self.training_mode and self.steps < int(self.initial_random_steps):
            action = self.action_space.sample()
        self.steps += 1
        self.time += self.dt
        turn_cmd = float(action[0]) * self.max_turn_rate
        speed_cmd = float(action[1]) * 2.0
        self.heading += turn_cmd * self.dt
        self.speed = np.clip(self.speed + speed_cmd * self.dt, self.min_speed, self.max_speed)
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        self.x = np.clip(self.x, self.world_bounds['x_min'], self.world_bounds['x_max'])
        self.y = np.clip(self.y, self.world_bounds['y_min'], self.world_bounds['y_max'])
        self.episode_positions.append((self.x, self.y))
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = truncated = False
        success = False
        episode_type = 'running'
        # Early success
        gd = self._goal_dist()
        if gd < self.goal_radius:
            terminated = True
            success = True
            episode_type = 'goal'
        elif self._in_collision() or self._out_of_bounds():
            terminated = True
            success = False
            episode_type = 'collision' if self._in_collision() else 'out_of_bounds'
        # LOS success: yol temiz ve hedef yakınsa başarı say
        elif self.allow_los_success and (not self._detect_cluster_obstacles()) and gd < (self.goal_radius * 1.8):
            terminated = True
            success = True
            episode_type = 'los_success'
        # Stuck detection: no meaningful improvement for patience steps
        if gd < self.best_goal_dist - 0.5:
            self.best_goal_dist = gd
            self.last_improve_step = self.steps
        else:
            # Patience increases near goal to avoid premature truncate
            patience = 200  # Daha fazla sabır - erken kesmeyi önle
            if gd < (self.goal_radius * 3.0):
                patience = 400  # Hedef yakınında daha da fazla sabır
            if (self.steps - self.last_improve_step) > patience and not terminated:
                # penalize and truncate to encourage learning shorter, purposeful paths
                reward -= 500.0
                truncated = True
        # Time limit
        if self.steps >= self.max_steps and not (terminated or truncated):
            truncated = True
            if self.allow_timeout_near_goal_success and gd < self.goal_radius * 1.5:
                success = True
                episode_type = 'timeout_near_goal'
        self.episode_rewards.append(reward)
        if terminated and success:
            self.success_count += 1
        info = {
            'goal_distance': gd,
            'collision': self._in_collision(),
            'out_of_bounds': self._out_of_bounds(),
            'episode_length': self.steps,
            'total_reward': sum(self.episode_rewards),
            'time': self.time,
            'success': success,
            'episode_type': episode_type if (terminated or truncated) else 'running'
        }
        return obs, reward, terminated, truncated, info

    # ----------- Helper Methods -----------
    def _predict_future_waypoints(self) -> np.ndarray:
        waypoints = []
        for t in [1.0, 2.0, 3.0]:
            future_x = self.x + self.speed * np.cos(self.heading) * t * 60
            future_y = self.y + self.speed * np.sin(self.heading) * t * 60
            waypoints.extend([future_x, future_y])
        return np.array(waypoints, dtype=np.float32)

    def _assess_waypoint_risk(self, waypoints: np.ndarray):
        total_risk = 0.0
        risky_clusters = 0
        for i in range(0, len(waypoints), 2):
            waypoint_x, waypoint_y = waypoints[i], waypoints[i+1]
            for cluster in self.clusters:
                d = np.hypot(cluster['cx'] - waypoint_x, cluster['cy'] - waypoint_y)
                if d < (cluster['r'] + self.safe_margin):
                    total_risk += cluster['risk']
                    risky_clusters += 1
        return total_risk, risky_clusters

    def _detect_cluster_obstacles(self):
        obstacles = []
        # Use current aircraft position to evaluate line-of-sight to goal (segment, not infinite line)
        start_x, start_y = self.x, self.y
        goal_x, goal_y = self.goal[0], self.goal[1]
        vx, vy = (goal_x - start_x), (goal_y - start_y)
        L2 = vx*vx + vy*vy
        if L2 < 1e-9:
            return obstacles
        for cluster in self.clusters:
            # Project cluster center onto segment to check the closest point along the path
            t = ((cluster['cx'] - start_x) * vx + (cluster['cy'] - start_y) * vy) / L2
            if t < 0.0:
                continue  # behind the aircraft; ignore
            if t > 1.0:
                t = 1.0
            px = start_x + t * vx
            py = start_y + t * vy
            distance_to_path = np.hypot(cluster['cx'] - px, cluster['cy'] - py)
            if distance_to_path < (cluster['r'] + self.safe_margin):
                obstacles.append(cluster)
        return obstacles

    def _detect_obstacles_between(self, start_x: float, start_y: float, goal_x: float, goal_y: float):
        obstacles = []
        if abs(goal_x - start_x) > 1e-6:
            m = (goal_y - start_y) / (goal_x - start_x)
            b = start_y - m * start_x
            for cluster in self.clusters:
                distance_to_line = abs(cluster['cy'] - m * cluster['cx'] - b) / np.sqrt(1 + m**2)
                if distance_to_line < (cluster['r'] + self.safe_margin):
                    if (min(start_x, goal_x) <= cluster['cx'] <= max(start_x, goal_x) and
                        min(start_y, goal_y) <= cluster['cy'] <= max(start_y, goal_y)):
                        obstacles.append(cluster)
        else:
            for cluster in self.clusters:
                if abs(cluster['cx'] - start_x) < (cluster['r'] + self.safe_margin):
                    if min(start_y, goal_y) <= cluster['cy'] <= max(start_y, goal_y):
                        obstacles.append(cluster)
        return obstacles

    def _calculate_safe_maneuver(self, cluster: dict):
        cx, cy, radius = cluster['cx'], cluster['cy'], cluster['r']
        # Define LOS vector and its normal from current position to goal
        vx, vy = (self.goal[0] - self.x), (self.goal[1] - self.y)
        L = np.hypot(vx, vy)
        if L < 1e-6:
            vx, vy, L = 1.0, 0.0, 1.0
        nx, ny = -vy / L, vx / L
        tx, ty = vx / L, vy / L

        base_offset = radius + self.safe_margin + 12.0

        def clearance_to_clusters(px: float, py: float) -> float:
            if not self.clusters:
                return float('inf')
            m = float('inf')
            for c in self.clusters:
                d_edge = np.hypot(c['cx'] - px, c['cy'] - py) - (c['r'] + self.safe_margin)
                if d_edge < m:
                    m = d_edge
            return m

        def path_is_clear(px: float, py: float) -> bool:
            return len(self._detect_obstacles_between(self.x, self.y, px, py)) == 0

        candidates = []
        for side in (+1.0, -1.0):
            for k in (1.0, 1.4):
                px = cx + side * nx * (base_offset * k) + tx * (radius + 6.0)
                py = cy + side * ny * (base_offset * k) + ty * (radius + 6.0)
                bx = np.clip(px, self.world_bounds['x_min'] + 5, self.world_bounds['x_max'] - 5)
                by = np.clip(py, self.world_bounds['y_min'] + 5, self.world_bounds['y_max'] - 5)
                candidates.append((bx, by))

        def score_candidate(pt):
            px, py = pt
            clear_path = path_is_clear(px, py)
            clear = clearance_to_clusters(px, py)
            dist_to_goal = -np.hypot(self.goal[0] - px, self.goal[1] - py)
            return (1 if clear_path else 0, clear, dist_to_goal)

        best = max(candidates, key=score_candidate)
        if score_candidate(best)[0] == 0:
            # If none has a clear leg, push further out along the normal direction
            more_px = cx + nx * (base_offset * 1.8) + tx * (radius + 10.0)
            more_py = cy + ny * (base_offset * 1.8) + ty * (radius + 10.0)
            best = (
                np.clip(more_px, self.world_bounds['x_min'] + 5, self.world_bounds['x_max'] - 5),
                np.clip(more_py, self.world_bounds['y_min'] + 5, self.world_bounds['y_max'] - 5),
            )
        return best

    def _is_maneuver_needed(self):
        return len(self._detect_cluster_obstacles()) > 0

    def _get_maneuver_target(self):
        obstacles = self._detect_cluster_obstacles()
        if not obstacles:
            return self.goal[0], self.goal[1]
        # Choose the obstacle earliest along the current LOS
        vx, vy = (self.goal[0] - self.x), (self.goal[1] - self.y)
        L = max(np.hypot(vx, vy), 1e-6)
        def along_distance(c):
            return ((c['cx'] - self.x) * vx + (c['cy'] - self.y) * vy) / L
        primary = min(obstacles, key=lambda c: max(along_distance(c), 0.0))
        target = self._calculate_safe_maneuver(primary)
        # Validate leg to target; if blocked and there is another obstacle, try that
        if len(self._detect_obstacles_between(self.x, self.y, target[0], target[1])) > 0 and len(obstacles) > 1:
            remaining = [o for o in obstacles if o is not primary]
            secondary = min(remaining, key=lambda c: max(along_distance(c), 0.0))
            target = self._calculate_safe_maneuver(secondary)
        return target

    # Active target for shaping: maneuver target if needed, otherwise goal
    def _active_target(self):
        if self._is_maneuver_needed():
            return self._get_maneuver_target()
        return self.goal[0], self.goal[1]

    # Allow curriculum phases to change env without recreation
    def set_curriculum(self, goal_radius=None, max_steps=None, curriculum_num_clusters=None, safe_margin=None):
        if goal_radius is not None:
            self.goal_radius = float(goal_radius)
        if max_steps is not None:
            self.max_steps = int(max_steps)
        if curriculum_num_clusters is not None:
            self.curriculum_num_clusters = int(curriculum_num_clusters)
        if safe_margin is not None:
            self.safe_margin = float(safe_margin)
        return True

    def set_observation_mode(self, mode: str = 'v1'):
        """Set observation mode. Must be called before training/env vectorization.
        'v1' => 30-dim, 'v2' => 36-dim (adds per-cluster dist/bearing for top-3 clusters).
        """
        assert mode in ('v1', 'v2'), "mode must be 'v1' or 'v2'"
        self.obs_mode = mode
        # Update observation_space only if switching before rollout starts
        if mode == 'v1':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        return True

    def _calculate_reward(self):
        goal_dist = self._goal_dist()
        # Time cost
        r_time = -2.0
        # Active target shaping
        ax, ay = self._active_target()
        r_progress = 0.0
        if len(self.episode_positions) > 1:
            prev_pos = self.episode_positions[-2]
            prev_dist = np.hypot(ax - prev_pos[0], ay - prev_pos[1])
            curr_dist = np.hypot(ax - self.x, ay - self.y)
            r_progress = (prev_dist - curr_dist) * 80.0
        # Heading alignment to active target
        target_angle = np.arctan2(ay - self.y, ax - self.x)
        bearing_err = target_angle - self.heading
        while bearing_err > np.pi: bearing_err -= 2*np.pi
        while bearing_err < -np.pi: bearing_err += 2*np.pi
        r_heading = 4.0 * np.cos(bearing_err)
        # Close-goal shaping
        r_close_goal = 100.0 if goal_dist < (self.goal_radius * 2.5) else 0.0
        # User-specified distance-to-goal shaping (small pull-in)
        r_dist_goal = -0.2 * goal_dist
        # Proactive via waypoint risk
        wps = self._predict_future_waypoints()
        wp_risk, wp_cnt = self._assess_waypoint_risk(wps)
        r_proactive = -wp_risk * 15.0 if wp_cnt > 0 else 60.0
        # Safe zone reward relative to nearest cluster edge
        if self.clusters:
            dmins = [np.hypot(c['cx'] - self.x, c['cy'] - self.y) - getattr(self, 'cluster_radius', c['r']) for c in self.clusters]
            r_safe_zone = 0.05 * max(dmins)
        else:
            r_safe_zone = 0.0
        # Continuous clearance penalty
        r_clear = 0.0
        for c in self.clusters:
            d = np.hypot(c['cx'] - self.x, c['cy'] - self.y)
            safe = c['r'] + self.safe_margin
            if d < safe:
                r_clear += -200.0 - 30.0 * (safe - d)
            elif d < safe + 10.0:
                r_clear += -(safe + 10.0 - d) * 3.0
        # Risk-shaped penalty (distance-weighted when inside safe zone)
        total_risk_penalty = 0.0
        for c in self.clusters:
            d = np.hypot(c['cx'] - self.x, c['cy'] - self.y)
            safe = c['r'] + self.safe_margin
            if d < safe:
                total_risk_penalty += (1.0 / max(d, 1e-2)) * (1.0 + c.get('risk', 0.5))
        r_risk = -15.0 * total_risk_penalty
        # Path clear bonus
        r_path_clear = 80.0 if not self._detect_cluster_obstacles() else 0.0
        # Subgoal proximity bonus
        if self._is_maneuver_needed() is False:
            r_subgoal = 0.0
        else:
            mx, my = self._get_maneuver_target()
            md = np.hypot(mx - self.x, my - self.y)
            r_subgoal = 60.0 if md < 12.0 else 0.0
        # Terminal terms
        r_goal = 15000.0 if goal_dist < self.goal_radius else 0.0
        r_collision = -1000.0 if self._in_collision() else 0.0
        r_bounds = -500.0 if self._out_of_bounds() else 0.0
        return (r_time + r_progress + r_heading + r_close_goal + r_dist_goal +
                r_proactive + r_safe_zone + r_clear + r_risk + r_path_clear + r_subgoal +
                r_goal + r_collision + r_bounds)

    def _get_obs(self):
        # Optional observation noise for robustness during training
        if self.training_mode and self.observation_noise_std > 0.0:
            noise_xy = np.random.normal(0.0, self.observation_noise_std, size=2)
            obs_x = float(self.x + noise_xy[0])
            obs_y = float(self.y + noise_xy[1])
            noise_goal = np.random.normal(0.0, self.observation_noise_std, size=2)
            goal_x_noisy = float(self.goal[0] + noise_goal[0])
            goal_y_noisy = float(self.goal[1] + noise_goal[1])
        else:
            obs_x, obs_y = float(self.x), float(self.y)
            goal_x_noisy, goal_y_noisy = float(self.goal[0]), float(self.goal[1])

        current_state = [obs_x, obs_y, self.speed, self.heading]
        predicted_waypoints = self._predict_future_waypoints()
        # Top 3 clusters raw
        cluster_info = []
        top_clusters = sorted(self.clusters, key=lambda c: np.hypot(c['cx'] - self.x, c['cy'] - self.y))[:3]
        for cluster in top_clusters:
            cluster_info.extend([cluster['cx'], cluster['cy'], cluster['r'], cluster['risk']])
        while len(cluster_info) < 12:
            cluster_info.extend([1000.0, 1000.0, 0.0, 0.0])
        # Goal info
        goal_info = [goal_x_noisy, goal_y_noisy, self._goal_dist(), self._goal_bearing()]
        # Maneuver info
        if self._is_maneuver_needed():
            target = self._get_maneuver_target()
            maneuver_info = [1.0, target[0], target[1], float(len(self._detect_cluster_obstacles()))]
        else:
            maneuver_info = [0.0, self.goal[0], self.goal[1], 0.0]
        if self.obs_mode == 'v2':
            # Cluster relative features (distance, bearing) for same top 3
            rel_features = []
            for cluster in top_clusters:
                dx, dy = cluster['cx'] - self.x, cluster['cy'] - self.y
                dist = float(np.hypot(dx, dy))
                bearing = float(np.arctan2(dy, dx) - self.heading)
                while bearing > np.pi: bearing -= 2*np.pi
                while bearing < -np.pi: bearing += 2*np.pi
                rel_features.extend([dist, bearing])
            while len(rel_features) < 6:
                rel_features.extend([1000.0, 0.0])
            obs_list = current_state + list(predicted_waypoints) + cluster_info + goal_info + maneuver_info + rel_features
        else:
            # v1: 30-dim without extra cluster-relative features
            obs_list = current_state + list(predicted_waypoints) + cluster_info + goal_info + maneuver_info
        return np.array(obs_list, dtype=np.float32)

    # ----- Training helpers -----
    def set_training_mode(self, enable: bool = True, initial_random_steps: int = 0):
        self.training_mode = bool(enable)
        self.initial_random_steps = int(max(0, initial_random_steps))
        return True

    def set_observation_noise(self, std: float = 0.0):
        self.observation_noise_std = float(max(0.0, std))
        return True

    def _goal_dist(self): return np.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
    def _goal_bearing(self): 
        relative_bearing = np.arctan2(self.goal[1]-self.y, self.goal[0]-self.x) - self.heading
        while relative_bearing > np.pi: relative_bearing -= 2*np.pi
        while relative_bearing < -np.pi: relative_bearing += 2*np.pi
        return relative_bearing
    def _in_collision(self): return any(np.hypot(c['cx'] - self.x, c['cy'] - self.y) < c['r'] for c in self.clusters)
    def _out_of_bounds(self): return (self.x < self.world_bounds['x_min'] or self.x > self.world_bounds['x_max'] or
                                      self.y < self.world_bounds['y_min'] or self.y > self.world_bounds['y_max'])
    def _sample_goal(self):
        largest_cluster = max(self.clusters, key=lambda c: c['r'])
        angle = random.uniform(0, 2*np.pi)
        distance = largest_cluster['r'] + random.uniform(20,40)
        goal_x = np.clip(largest_cluster['cx'] + distance*np.cos(angle), self.world_bounds['x_min']+10, self.world_bounds['x_max']-10)
        goal_y = np.clip(largest_cluster['cy'] + distance*np.sin(angle), self.world_bounds['y_min']+10, self.world_bounds['y_max']-10)
        return np.array([goal_x, goal_y], dtype=np.float32)
    def _sample_clusters(self):
        clusters = []
        num_clusters = self.curriculum_num_clusters if getattr(self, 'curriculum_num_clusters', 0) > 0 else random.randint(1,3)
        rad = getattr(self, 'cluster_radius', None)
        for i in range(num_clusters):
            while True:
                cx, cy = random.uniform(20,120), random.uniform(20,120)
                if all(np.hypot(cx - c['cx'], cy - c['cy']) > 25 for c in clusters): break
            r = float(rad) if rad is not None else random.uniform(5,10)
            clusters.append({'cx': cx, 'cy': cy, 'r': r, 'risk': random.uniform(0.4,0.7), 'id': i})
        return clusters

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12,10))
        self.ax.clear()
        for cluster in self.clusters:
            self.ax.add_patch(Circle((cluster['cx'], cluster['cy']), cluster['r'], color='red', alpha=0.3))
            self.ax.add_patch(Circle((cluster['cx'], cluster['cy']), cluster['r']+self.safe_margin, color='orange', alpha=0.1))
        waypoints = self._predict_future_waypoints()
        # Waypoint'ler arasında çizgi çek
        if len(waypoints) >= 4:
            wp_x = [waypoints[i] for i in range(0, len(waypoints), 2)]
            wp_y = [waypoints[i+1] for i in range(0, len(waypoints), 2)]
            self.ax.plot(wp_x, wp_y, 'r--', alpha=0.4, linewidth=2)  # Kırmızı kesikli çizgi
        
        for i in range(0,len(waypoints),2):
            # Farklı zamanlardaki waypoint'ler için farklı boyutlar
            time_factor = (i // 2 + 1) * 0.5
            size = int(80 + time_factor * 40)
            alpha = 0.6 + time_factor * 0.2
            self.ax.scatter(waypoints[i], waypoints[i+1], c='red', s=size, marker='o', alpha=alpha, edgecolors='darkred', linewidth=1)
        self.ax.scatter(self.goal[0], self.goal[1], c='green', s=200, marker='*')
        self.ax.scatter(self.x, self.y, c='blue', s=150, marker='^')
        arrow_len = 15
        self.ax.arrow(self.x, self.y, arrow_len*np.cos(self.heading), arrow_len*np.sin(self.heading),
                      head_width=3, head_length=5, fc='blue', ec='blue')
        if len(self.episode_positions) > 1:
            traj_x, traj_y = zip(*self.episode_positions)
            self.ax.plot(traj_x, traj_y, 'b-', alpha=0.7, linewidth=2)
        self.ax.set_xlim(self.world_bounds['x_min'], self.world_bounds['x_max'])
        self.ax.set_ylim(self.world_bounds['y_min'], self.world_bounds['y_max'])
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'Proaktif Navigation - Zaman: {self.time:.1f}s')
        self.ax.grid(True, alpha=0.3); plt.tight_layout(); plt.pause(0.01)

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig); self.fig = None; self.ax = None

# ----------- Test -----------
def test_proactive_environment():
    env = ProactiveNavigationEnv()
    obs,_ = env.reset()
    print("Initial Observation:", obs)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print("Episode finished:", info)
            break
    env.close()

if __name__ == "__main__":
    test_proactive_environment()
