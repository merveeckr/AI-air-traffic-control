"""
Canlı AI Navigation Simülasyonu
AI Agent'ın proaktif manevralarını gerçek zamanlı göster
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from proactive_navigation_env import ProactiveNavigationEnv
from proactive_training import ProactiveNavigationTrainer

class LiveAISimulation:
    """Canlı AI simülasyonu"""
    
    def __init__(self):
        self.env = ProactiveNavigationEnv()
        self.trainer = ProactiveNavigationTrainer()
        
        # Eğitilmiş modeli yükle
        try:
            self.model = self.trainer.load_model()
            print("✅ Eğitilmiş model yüklendi!")
        except:
            print("❌ Model bulunamadı, yeni training başlatılıyor...")
            self.model = self.trainer.train_model()
        
        # Görselleştirme ayarları
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.ion()  # Interactive mode
        
    def run_simulation(self, num_episodes=3, slow_motion=True):
        """Canlı simülasyon çalıştır"""
        print(f"🚀 {num_episodes} Episode Canlı Simülasyon Başlıyor...")
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            obs, info = self.env.reset()
            total_reward = 0
            step_count = 0
            
            # Episode boyunca simülasyon
            while True:
                # AI action'ı al
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # Her 5 adımda görselleştir (daha akıcı)
                if step_count % 5 == 0:
                    self._visualize_step(episode, step_count, total_reward, info)
                    
                    if slow_motion:
                        time.sleep(0.1)  # 0.1 saniye bekle
                    
                    plt.pause(0.01)
                
                # Episode durumu kontrol et
                if terminated or truncated:
                    break
            
            # Episode sonucu
            print(f"Episode {episode + 1} Tamamlandı:")
            print(f"  Steps: {step_count}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Success: {info.get('success', False)}")
            print(f"  Final Goal Distance: {self.env._goal_dist():.1f}")
            
            # Son durumu göster
            self._visualize_final_state(episode, info)
            plt.pause(2)  # 2 saniye bekle
        
        plt.ioff()
        plt.show()
        print("\n🎉 Simülasyon tamamlandı!")
    
    def _visualize_step(self, episode, step, total_reward, info):
        """Her adımı görselleştir"""
        self.ax.clear()
        
        # Kümeleri çiz
        for cluster in self.env.clusters:
            # Ana küme
            circle = Circle((cluster['cx'], cluster['cy']), cluster['r'], 
                          color='red', alpha=0.4, label=f"Küme {cluster['id']}")
            self.ax.add_patch(circle)
            
            # Güvenlik tamponu
            safe_circle = Circle((cluster['cx'], cluster['cy']), 
                               cluster['r'] + self.env.safe_margin, 
                               color='orange', alpha=0.2, linestyle='--')
            self.ax.add_patch(safe_circle)
        
        # Tahmin edilen waypoint'leri çiz
        waypoints = self.env._predict_future_waypoints()
        for i in range(0, len(waypoints), 2):
            wp_x, wp_y = waypoints[i], waypoints[i+1]
            time_min = (i // 2 + 1)
            
            # Waypoint risk kontrolü
            risk, risky_count = self.env._assess_waypoint_risk(waypoints)
            if risky_count > 0:
                color = 'red'  # Riskli waypoint
                size = 150
            else:
                color = 'green'  # Güvenli waypoint
                size = 100
            
            self.ax.scatter(wp_x, wp_y, c=color, s=size, marker='s', 
                           label=f'{time_min}dk Sonra', alpha=0.8)
        
        # Hedef
        self.ax.scatter(self.env.goal[0], self.env.goal[1], c='green', s=300, marker='*', 
                       label='Hedef', zorder=5)
        
        # Uçak
        self.ax.scatter(self.env.x, self.env.y, c='blue', s=200, marker='^', 
                       label='AI Uçak', zorder=5)
        
        # Uçak yönü (ok)
        arrow_length = 20
        arrow_x = self.env.x + arrow_length * np.cos(self.env.heading)
        arrow_y = self.env.y + arrow_length * np.sin(self.env.heading)
        self.ax.arrow(self.env.x, self.env.y, arrow_x - self.env.x, arrow_y - self.env.y, 
                     head_width=4, head_length=6, fc='blue', ec='blue', linewidth=2)
        
        # Trajectory (rota)
        if len(self.env.episode_positions) > 1:
            traj_x = [p[0] for p in self.env.episode_positions]
            traj_y = [p[1] for p in self.env.episode_positions]
            self.ax.plot(traj_x, traj_y, 'b-', alpha=0.8, linewidth=3, label='AI Rotası')
        
        # Harita ayarları
        self.ax.set_xlim(self.env.world_bounds['x_min'], self.env.world_bounds['x_max'])
        self.ax.set_ylim(self.env.world_bounds['y_min'], self.env.world_bounds['y_max'])
        self.ax.set_xlabel('X Koordinatı (m)', fontsize=12)
        self.ax.set_ylabel('Y Koordinatı (m)', fontsize=12)
        
        # Başlık ve bilgiler
        title = f'Episode {episode + 1} - Step {step} - AI Proaktif Navigation'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Waypoint risk bilgisi
        risk, risky_count = self.env._assess_waypoint_risk(waypoints)
        risk_text = f'Waypoint Risk: {risk:.2f}, Risky Clusters: {risky_count}'
        self.ax.text(0.02, 0.98, risk_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Hedef mesafesi
        goal_dist = self.env._goal_dist()
        dist_text = f'Goal Distance: {goal_dist:.1f}m'
        self.ax.text(0.02, 0.92, dist_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Layout
        plt.tight_layout()
    
    def _visualize_final_state(self, episode, info):
        """Episode son durumunu göster"""
        self.ax.set_title(f'Episode {episode + 1} Tamamlandı - {info.get("episode_type", "unknown")}', 
                         fontsize=16, fontweight='bold')
        
        # Başarı durumu
        if info.get('success', False):
            status_color = 'green'
            status_text = '🎉 BAŞARILI!'
        else:
            status_color = 'red'
            status_text = '❌ Başarısız'
        
        self.ax.text(0.5, 0.5, status_text, transform=self.ax.transAxes, 
                    fontsize=20, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))

def main():
    """Ana fonksiyon"""
    print("🚁 AI Proaktif Navigation Canlı Simülasyonu")
    print("=" * 50)
    
    # Simülasyon başlat
    sim = LiveAISimulation()
    
    # Kullanıcı seçimi
    print("\nSimülasyon seçenekleri:")
    print("1. Hızlı simülasyon (3 episode)")
    print("2. Yavaş simülasyon (3 episode, slow motion)")
    print("3. Tek episode detaylı")
    
    choice = input("Seçiminiz (1-3): ").strip()
    
    if choice == "1":
        sim.run_simulation(num_episodes=3, slow_motion=False)
    elif choice == "2":
        sim.run_simulation(num_episodes=3, slow_motion=True)
    elif choice == "3":
        sim.run_simulation(num_episodes=1, slow_motion=True)
    else:
        print("Geçersiz seçim, varsayılan başlatılıyor...")
        sim.run_simulation(num_episodes=2, slow_motion=True)

if __name__ == "__main__":
    main()
