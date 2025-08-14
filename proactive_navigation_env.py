"""
Proaktif Aircraft Navigation Environment
Waypoint tahmini + proaktif risk kaçınma için özel ortam
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from typing import Tuple, Dict, Any, Optional

class ProactiveNavigationEnv(gym.Env):
    """Proaktif navigasyon ortamı - Waypoint tahmini + manevra planlama"""
    
    def __init__(self, prediction_horizon=3.0, dt=0.1, max_steps=2000, render_mode=None):
        super().__init__()
        self.prediction_horizon = prediction_horizon  # 3 dakika tahmin
        self.dt = dt  # Zaman adımı (0.1 saniye)
        self.max_steps = max_steps
        self.max_turn_rate = np.deg2rad(30)  # Maksimum dönüş hızı
        self.max_speed = 20.0  # Maksimum hız (m/s)
        self.min_speed = 10.0  # Minimum hız (m/s)
        
        # Güvenlik parametreleri - Daha kolay hale getir
        self.safe_margin = 15.0  # Güvenlik tamponu (25.0'dan 15.0'a)
        self.goal_radius = 25.0  # Hedef yarıçapı (15.0'dan 25.0'a)
        self.world_bounds = {'x_min': -50, 'x_max': 150, 'y_min': -50, 'y_max': 150}
        
        # Observation space: [current_state, predicted_waypoints, cluster_info, goal_info]
        # current_state: [x, y, speed, heading] = 4
        # predicted_waypoints: [x1, y1, x2, y2, x3, y3] = 6 (3 waypoint)
        # cluster_info: [cluster1_x, cluster1_y, cluster1_r, cluster1_risk, ...] = 12 (3 cluster)
        # goal_info: [goal_x, goal_y, goal_distance, goal_bearing] = 4
        # Total: 4 + 6 + 12 + 4 = 26
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        
        # Action space: [delta_heading_rate, delta_speed] (sürekli)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        
        # Ortam durumu
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ortamı sıfırla"""
        super().reset(seed=seed)
        
        self.steps = 0
        self.time = 0.0  # Geçen süre (saniye)
        
        # Uçağı başlangıç pozisyonuna yerleştir
        self.x = 0.0
        self.y = 0.0
        self.speed = 15.0  # m/s
        self.heading = 0.0  # radyan
        
        # Hedef ve kümeleri oluştur
        self.goal = self._sample_goal()
        self.clusters = self._sample_clusters()
        
        # Episode istatistikleri
        self.episode_rewards = []
        self.episode_positions = [(self.x, self.y)]
        self.episode_times = [0.0]
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Bir adım ilerle"""
        self.steps += 1
        self.time += self.dt
        
        # Action'ları ölçekle
        turn_cmd = float(action[0]) * self.max_turn_rate
        speed_cmd = float(action[1]) * 2.0  # ±2 m/s hız değişimi
        
        # Uçak dinamiğini güncelle
        self.heading += turn_cmd * self.dt
        self.speed = np.clip(self.speed + speed_cmd * self.dt, self.min_speed, self.max_speed)
        
        # Pozisyonu güncelle
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        
        # Sınırları kontrol et
        self.x = np.clip(self.x, self.world_bounds['x_min'], self.world_bounds['x_max'])
        self.y = np.clip(self.y, self.world_bounds['y_min'], self.world_bounds['y_max'])
        
        # Trajectory'yi kaydet
        self.episode_positions.append((self.x, self.y))
        self.episode_times.append(self.time)
        
        # Gözlem ve ödül hesapla
        obs = self._get_obs()
        reward = self._calculate_reward()
        
        # Episode durumu
        terminated = False
        truncated = False
        success = False  # Başarı durumu
        
        # Başarı kontrolü - Daha esnek
        goal_dist = self._goal_dist()
        if goal_dist < self.goal_radius:
            terminated = True
            success = True  # ✅ Hedefe ulaştı = Başarı!
        
        # Çarpışma kontrolü
        if self._in_collision():
            terminated = True
            success = False  # ❌ Çarpışma = Başarısızlık
        
        # Adım sınırı
        if self.steps >= self.max_steps:
            truncated = True
            # Time limit'te başarı kontrolü
            if goal_dist < self.goal_radius * 1.5:  # 1.5x tolerans
                success = True  # ✅ Yakınsa başarı say
        
        # Sınır dışı çıkma
        if self._out_of_bounds():
            terminated = True
            success = False  # ❌ Sınır dışı = Başarısızlık
        
        # İstatistikleri güncelle
        self.episode_rewards.append(reward)
        
        info = {
            'goal_distance': goal_dist,
            'collision': self._in_collision(),
            'out_of_bounds': self._out_of_bounds(),
            'episode_length': self.steps,
            'total_reward': sum(self.episode_rewards),
            'time': self.time,
            'success': success,  # ✅ Başarı durumu eklendi!
            'episode_type': 'success' if success else ('collision' if self._in_collision() else 'timeout')
        }
        
        return obs, reward, terminated, truncated, info
    
    def _predict_future_waypoints(self) -> np.ndarray:
        """Gelecek waypoint'leri tahmin et (AI burada devreye girecek)"""
        # Basit lineer tahmin (gerçek sistemde LSTM/Transformer kullanılacak)
        waypoints = []
        
        for t in [1.0, 2.0, 3.0]:  # 1dk, 2dk, 3dk sonra
            # Mevcut hız ve yöne göre basit tahmin
            future_x = self.x + self.speed * np.cos(self.heading) * t * 60  # dakika -> saniye
            future_y = self.y + self.speed * np.sin(self.heading) * t * 60
            
            waypoints.extend([future_x, future_y])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _assess_waypoint_risk(self, waypoints: np.ndarray) -> Tuple[float, int]:
        """Waypoint'lerin risk seviyesini değerlendir"""
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
    
    def _calculate_reward(self) -> float:
        """Ödül hesapla - Proaktif yaklaşım"""
        goal_dist = self._goal_dist()
        
        # 1. Hedefe yaklaşma ödülü - Daha büyük ödül
        r_progress = 0.0
        if len(self.episode_positions) > 1:
            prev_pos = self.episode_positions[-2]
            prev_dist = np.hypot(self.goal[0] - prev_pos[0], self.goal[1] - prev_pos[1])
            r_progress = (prev_dist - goal_dist) * 20.0  # 10.0'dan 20.0'a
        
        # 2. Proaktif risk kaçınma ödülü
        waypoints = self._predict_future_waypoints()
        waypoint_risk, risky_count = self._assess_waypoint_risk(waypoints)
        
        if risky_count > 0:
            r_proactive = -waypoint_risk * 3.0  # 5.0'dan 3.0'a (daha yumuşak ceza)
        else:
            r_proactive = 20.0  # 10.0'dan 20.0'a (daha büyük ödül)
        
        # 3. Hedefe ulaşma ödülü - Çok daha büyük ödül
        r_goal = 0.0
        if goal_dist < self.goal_radius:
            r_goal = 5000.0  # 1000.0'dan 5000.0'a
        
        # 4. Çarpışma cezası - Daha yumuşak ceza
        r_collision = 0.0
        if self._in_collision():
            r_collision = -200.0  # -500.0'dan -200.0'a
        
        # 5. Sınır dışı cezası - Daha yumuşak ceza
        r_bounds = 0.0
        if self._out_of_bounds():
            r_bounds = -100.0  # -200.0'dan -100.0'a
        
        # Toplam ödül
        total_reward = r_progress + r_proactive + r_goal + r_collision + r_bounds
        
        return total_reward
    
    def _get_obs(self) -> np.ndarray:
        """Gözlem vektörünü oluştur"""
        # 1. Mevcut durum
        current_state = [self.x, self.y, self.speed, self.heading]
        
        # 2. Tahmin edilen waypoint'ler
        predicted_waypoints = self._predict_future_waypoints()
        
        # 3. Küme bilgileri (en yakın 3 küme)
        cluster_info = []
        distances = []
        
        for cluster in self.clusters:
            d = np.hypot(cluster['cx'] - self.x, cluster['cy'] - self.y)
            distances.append((d, cluster))
        
        distances.sort(key=lambda x: x[0])
        top_clusters = [c for _, c in distances[:3]]
        
        for cluster in top_clusters:
            cluster_info.extend([cluster['cx'], cluster['cy'], cluster['r'], cluster['risk']])
        
        # Eğer 3'ten az küme varsa pad et
        while len(cluster_info) < 12:
            cluster_info.extend([1000.0, 1000.0, 0.0, 0.0])
        
        # 4. Hedef bilgileri
        goal_info = [
            self.goal[0],                    # Hedef X
            self.goal[1],                    # Hedef Y
            self._goal_dist(),               # Hedef mesafesi
            self._goal_bearing()             # Hedef açısı
        ]
        
        # Tüm gözlemleri birleştir
        obs = np.array(
            current_state + list(predicted_waypoints) + cluster_info + goal_info,
            dtype=np.float32
        )
        
        return obs
    
    def _goal_dist(self) -> float:
        """Hedefe olan mesafe"""
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        return np.hypot(dx, dy)
    
    def _goal_bearing(self) -> float:
        """Hedefe olan açı (heading'e göre)"""
        target_bearing = np.arctan2(self.goal[1] - self.y, self.goal[0] - self.x)
        relative_bearing = target_bearing - self.heading
        
        # Açıyı [-π, π] aralığına normalize et
        while relative_bearing > np.pi:
            relative_bearing -= 2 * np.pi
        while relative_bearing < -np.pi:
            relative_bearing += 2 * np.pi
        
        return relative_bearing
    
    def _in_collision(self) -> bool:
        """Çarpışma kontrolü"""
        for cluster in self.clusters:
            d = np.hypot(cluster['cx'] - self.x, cluster['cy'] - self.y)
            if d < (cluster['r'] + self.safe_margin):
                return True
        return False
    
    def _out_of_bounds(self) -> bool:
        """Sınır dışı kontrolü"""
        return (self.x < self.world_bounds['x_min'] or 
                self.x > self.world_bounds['x_max'] or
                self.y < self.world_bounds['y_min'] or 
                self.y > self.world_bounds['y_max'])
    
    def _sample_goal(self) -> np.ndarray:
        """Hedef nokta örneği - Daha yakın hedefler"""
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(30, 60)  # 80-120'den 30-60'a (çok daha yakın!)
        
        goal_x = distance * np.cos(angle)
        goal_y = distance * np.sin(angle)
        
        return np.array([goal_x, goal_y], dtype=np.float32)
    
    def _sample_clusters(self) -> list:
        """Küme örnekleri oluştur - Daha uzak yerleştir"""
        clusters = []
        num_clusters = random.randint(1, 2)  # 1-3'ten 1-2'ye (daha az küme)
        
        for i in range(num_clusters):
            attempts = 0
            while attempts < 100:  # 50'den 100'e
                cx = random.uniform(self.world_bounds['x_min'] + 50, self.world_bounds['x_max'] - 50)  # 30'dan 50'ye
                cy = random.uniform(self.world_bounds['y_min'] + 50, self.world_bounds['y_max'] - 50)  # 30'dan 50'ye
                
                # Başlangıç noktasından çok uzak olsun
                start_dist = np.hypot(cx, cy)
                if start_dist < 60:  # 30'dan 60'a (çok daha uzak)
                    attempts += 1
                    continue
                
                # Hedef yolunda olmamasına dikkat et
                goal_dist = np.hypot(cx - self.goal[0], cy - self.goal[1])
                if goal_dist < 60:  # 40'dan 60'a (çok daha uzak)
                    attempts += 1
                    continue
                
                break
            
            # Küme özellikleri - Daha da küçük ve daha az riskli
            radius = random.uniform(3, 8)  # 5-12'den 3-8'e (çok daha küçük kümeler)
            risk = random.uniform(0.1, 0.5)  # 0.3-0.8'den 0.1-0.5'e (çok daha az risk)
            
            clusters.append({
                'cx': cx,
                'cy': cy,
                'r': radius,
                'risk': risk,
                'id': i
            })
        
        return clusters
    
    def render(self, mode='human'):
        """Ortamı görselleştir"""
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.ax.clear()
        
        # Kümeleri çiz
        for cluster in self.clusters:
            circle = Circle((cluster['cx'], cluster['cy']), cluster['r'], 
                          color='red', alpha=0.3, label=f"Küme {cluster['id']}")
            self.ax.add_patch(circle)
            
            # Güvenlik tamponu
            safe_circle = Circle((cluster['cx'], cluster['cy']), 
                               cluster['r'] + self.safe_margin, 
                               color='orange', alpha=0.1, linestyle='--')
            self.ax.add_patch(safe_circle)
        
        # Tahmin edilen waypoint'leri çiz
        waypoints = self._predict_future_waypoints()
        for i in range(0, len(waypoints), 2):
            wp_x, wp_y = waypoints[i], waypoints[i+1]
            time_min = (i // 2 + 1)
            self.ax.scatter(wp_x, wp_y, c='purple', s=100, marker='s', 
                           label=f'{time_min}dk Sonra', alpha=0.7)
        
        # Hedef
        self.ax.scatter(self.goal[0], self.goal[1], c='green', s=200, marker='*', 
                       label='Hedef', zorder=5)
        
        # Uçak
        self.ax.scatter(self.x, self.y, c='blue', s=150, marker='^', 
                       label='Uçak', zorder=5)
        
        # Uçak yönü
        arrow_length = 15
        arrow_x = self.x + arrow_length * np.cos(self.heading)
        arrow_y = self.y + arrow_length * np.sin(self.heading)
        self.ax.arrow(self.x, self.y, arrow_x - self.x, arrow_y - self.y, 
                     head_width=3, head_length=5, fc='blue', ec='blue')
        
        # Trajectory
        if len(self.episode_positions) > 1:
            traj_x = [p[0] for p in self.episode_positions]
            traj_y = [p[1] for p in self.episode_positions]
            self.ax.plot(traj_x, traj_y, 'b-', alpha=0.7, linewidth=2, label='Rota')
        
        # Harita ayarları
        self.ax.set_xlim(self.world_bounds['x_min'], self.world_bounds['x_max'])
        self.ax.set_ylim(self.world_bounds['y_min'], self.world_bounds['y_max'])
        self.ax.set_xlabel('X Koordinatı (m)')
        self.ax.set_ylabel('Y Koordinatı (m)')
        self.ax.set_title(f'Proaktif Navigation - Zaman: {self.time:.1f}s')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self):
        """Ortamı kapat"""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

def test_proactive_environment():
    """Proaktif ortamı test et"""
    print("🚀 Proaktif Navigation Environment Test Ediliyor...")
    
    env = ProactiveNavigationEnv()
    
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Birkaç random adım test et
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
            print(f"Position: ({env.x:.1f}, {env.y:.1f}), Goal Distance: {env._goal_dist():.1f}")
            
            # Waypoint risk analizi
            waypoints = env._predict_future_waypoints()
            risk, count = env._assess_waypoint_risk(waypoints)
            print(f"Waypoint Risk: {risk:.2f}, Risky Clusters: {count}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Final Total Reward: {total_reward:.2f}")
    
    # Görselleştir
    env.render()
    plt.show()
    
    env.close()

if __name__ == "__main__":
    test_proactive_environment()
