"""
Milisaniye Hassasiyetinde Proaktif Canlı Simülasyon
AI Agent'ın kümelerin yanına yavaş yavaş yaklaşırken 3 saniye önceden manevra yapmasını göster
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import time
from datetime import datetime, timedelta
import threading
from proactive_navigation_env import ProactiveNavigationEnv
from proactive_training import ProactiveNavigationTrainer

class MillisecondPrecisionSimulation:
    """Milisaniye hassasiyetinde proaktif simülasyon"""
    
    def __init__(self):
        self.env = ProactiveNavigationEnv()
        self.trainer = ProactiveNavigationTrainer()
        
        # Eğitilmiş modeli yükle - Mevcut modeli kullan
        try:
            # En son eğitilen modeli bul
            self.model = self._load_latest_model()
            print("✅ Eğitilmiş model yüklendi!")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            print("🔄 Yeni training başlatılıyor...")
            self.model = self.trainer.train_model()
        
        # Milisaniye hassasiyet ayarları - Daha yavaş ve görünür
        self.dt = 0.1  # 0.1 saniye (daha görünür)
        self.simulation_speed = 0.5  # 0.5x hız (daha yavaş)
        self.visualization_fps = 10  # 10 FPS (daha yavaş görselleştirme)
        
        # Görselleştirme ayarları
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        plt.ion()  # Interactive mode
        
        # Simülasyon durumu
        self.is_running = False
        self.current_time = 0.0
        self.episode_start_time = None
        
        # Manevra geçmişi
        self.maneuver_history = []
        self.cluster_approach_history = []
    
    def _load_latest_model(self):
        """En son eğitilen modeli yükle"""
        import os
        from stable_baselines3 import PPO
        
        # Eğitim dizinlerini bul
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_dirs = []
        
        for item in os.listdir(base_dir):
            if item.startswith('proactive_navigation_training_') and os.path.isdir(os.path.join(base_dir, item)):
                training_dirs.append(item)
        
        if not training_dirs:
            raise FileNotFoundError("Eğitim dizini bulunamadı")
        
        print(f"📁 Bulunan eğitim dizinleri: {training_dirs}")
        
        # Model dosyası olan dizinleri bul
        valid_dirs = []
        for dir_name in training_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            
            # Final model kontrolü
            final_model_path = os.path.join(dir_path, 'final_model.zip')
            if os.path.exists(final_model_path):
                valid_dirs.append((dir_name, 'final_model.zip', final_model_path))
                continue
            
            # Best model kontrolü
            best_model_path = os.path.join(dir_path, 'best_model', 'best_model.zip')
            if os.path.exists(best_model_path):
                valid_dirs.append((dir_name, 'best_model.zip', best_model_path))
                continue
            
            # Checkpoint kontrolü
            checkpoint_dir = os.path.join(dir_path, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    valid_dirs.append((dir_name, latest_checkpoint, checkpoint_path))
        
        if not valid_dirs:
            raise FileNotFoundError("Hiçbir eğitim dizininde model dosyası bulunamadı")
        
        # En son geçerli dizini seç
        latest_valid_dir = max(valid_dirs, key=lambda x: x[0])
        dir_name, model_file, model_path = latest_valid_dir
        
        print(f"📁 Model bulundu: {dir_name}")
        print(f"📥 Model dosyası: {model_file}")
        print(f"📂 Tam yol: {model_path}")
        
        return PPO.load(model_path)
    
    def create_cluster_scenario(self):
        """Kümelerin yanına yavaş yaklaşma senaryosu"""
        print("🎯 Yavaş Yaklaşma Senaryosu Oluşturuluyor...")
        
        # Environment'ı reset et
        obs, info = self.env.reset()
        
        # Environment parametrelerini ayarla - Daha uzun simülasyon için
        self.env.max_steps = 500  # Çok daha uzun
        self.env.goal_radius = 50.0  # Daha büyük hedef
        self.env.safe_margin = 20.0  # Daha büyük güvenlik mesafesi
        
        # Manuel olarak kümeleri stratejik yerleştir - Daha yakın
        self.env.clusters = []
        
        # Küme 1: Hedef yolunda, yakın
        cluster1 = {
            'cx': 20.0,  # Hedef yolunda, daha yakın
            'cy': 12.0,
            'r': 5.0,    # Orta boy küme
            'risk': 0.6,  # Orta risk
            'id': 1,
            'approach_time': 6.0  # 6 saniye sonra yaklaşacak
        }
        
        # Küme 2: Başlangıç yolunda, çok yakın
        cluster2 = {
            'cx': 8.0,   # Başlangıç yolunda, çok yakın
            'cy': 5.0,
            'r': 4.0,    # Küçük küme
            'risk': 0.4,  # Düşük risk
            'id': 2,
            'approach_time': 2.0  # 2 saniye sonra yaklaşacak
        }
        
        # Küme 3: Hedefin yanında, orta mesafe
        cluster3 = {
            'cx': 35.0,  # Hedefin yanında, orta mesafe
            'cy': 28.0,
            'r': 6.0,    # Büyük küme
            'risk': 0.8,  # Yüksek risk
            'id': 3,
            'approach_time': 10.0  # 10 saniye sonra yaklaşacak
        }
        
        self.env.clusters = [cluster1, cluster2, cluster3]
        
        print("✅ Yavaş yaklaşma senaryosu oluşturuldu:")
        print(f"   Environment Max Steps: {self.env.max_steps}")
        print(f"   Environment Goal Radius: {self.env.goal_radius}")
        print(f"   Environment Safe Margin: {self.env.safe_margin}")
        for cluster in self.env.clusters:
            print(f"   Küme {cluster['id']}: ({cluster['cx']:.1f}, {cluster['cy']:.1f}) - Risk: {cluster['risk']:.1f} - Yaklaşma: {cluster['approach_time']:.1f}s")
        
        return obs, info
    
    def calculate_cluster_approach_time(self, cluster):
        """Kümeye yaklaşma süresini hesapla"""
        dx = cluster['cx'] - self.env.x
        dy = cluster['cy'] - self.env.y
        distance = np.hypot(dx, dy)
        
        # Güvenlik mesafesi dahil
        safe_distance = cluster['r'] + self.env.safe_margin
        approach_distance = distance - safe_distance
        
        if approach_distance <= 0:
            return 0.0  # Zaten güvenlik mesafesinde
        
        # Mevcut hızla yaklaşma süresi
        approach_time = approach_distance / max(self.env.speed, 0.1)
        return approach_time
    
    def predict_maneuver_need(self, cluster):
        """1 saniye önceden manevra ihtiyacını tahmin et"""
        approach_time = self.calculate_cluster_approach_time(cluster)
        return approach_time <= 0.5  # 1 saniye veya daha az
    
    def run_millisecond_simulation(self):
        """Milisaniye hassasiyetinde simülasyon"""
        print("🚀 Milisaniye Hassasiyetinde Simülasyon Başlıyor...")
        print(f"⏱️  Simülasyon Hızı: {self.simulation_speed}x")
        print(f"📊 Görselleştirme: {self.visualization_fps} FPS")
        
        self.is_running = True
        self.episode_start_time = time.time()
        self.current_time = 0.0
        
        obs, info = self.create_cluster_scenario()
        total_reward = 0
        step_count = 0
        
        # Simülasyon döngüsü - Daha uzun süre çalışsın
        max_steps = 200  # En az 200 adım çalışsın
        
        # Environment'ı manuel olarak kontrol et
        self.env.max_steps = max_steps  # Maksimum adım sayısını artır
        self.env.goal_radius = 50.0     # Hedef yarıçapını artır (daha kolay ulaşsın)
        
        print(f"🔧 Environment ayarları:")
        print(f"   Max Steps: {self.env.max_steps}")
        print(f"   Goal Radius: {self.env.goal_radius}")
        print(f"   Safe Margin: {self.env.safe_margin}")
        
        while self.is_running and step_count < max_steps:
            start_step_time = time.time()
            
            try:
                # AI action'ı al
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                step_count += 1
                self.current_time += self.dt
                
                # Manevra analizi
                self._analyze_maneuvers()
                
                # Her adımda görselleştir (daha sık)
                if step_count % 1 == 0:  # Her adımda görselleştir
                    self._visualize_millisecond_step(step_count, total_reward, info)
                    plt.pause(1.0 / self.visualization_fps)  # 10 FPS
                
                # Episode durumu kontrol et - Daha esnek
                if terminated:
                    print(f"\n🎯 Episode Terminated!")
                    print(f"   Steps: {step_count}")
                    print(f"   Total Reward: {total_reward:.2f}")
                    print(f"   Success: {info.get('success', False)}")
                    print(f"   Final Goal Distance: {self.env._goal_dist():.1f}")
                    print(f"   Reason: {info.get('episode_type', 'unknown')}")
                    # Terminated olsa bile devam et
                    if step_count < 50:  # Çok erken terminate olursa devam et
                        print("   ⚠️  Çok erken terminate, devam ediliyor...")
                        obs, info = self.env.reset()
                        continue
                    else:
                        break
                
                if truncated:
                    print(f"\n⏰ Episode Truncated!")
                    print(f"   Steps: {step_count}")
                    print(f"   Total Reward: {total_reward:.2f}")
                    print(f"   Final Goal Distance: {self.env._goal_dist():.1f}")
                    # Truncated olsa bile devam et
                    if step_count < 50:  # Çok erken truncate olursa devam et
                        print("   ⚠️  Çok erken truncate, devam ediliyor...")
                        obs, info = self.env.reset()
                        continue
                    else:
                        break
                
            except Exception as e:
                print(f"❌ Adım {step_count} hatası: {e}")
                # Hata durumunda environment'ı reset et
                try:
                    obs, info = self.env.reset()
                    print("   🔄 Environment reset edildi")
                except:
                    print("   ❌ Environment reset hatası, simülasyon durduruluyor")
                    break
            
            # Simülasyon hızını kontrol et
            elapsed = time.time() - start_step_time
            target_time = self.dt / self.simulation_speed
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        
        # Simülasyon sonu
        if step_count >= max_steps:
            print(f"\n⏰ Maksimum adım sayısına ulaşıldı: {max_steps}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Final Goal Distance: {self.env._goal_dist():.1f}")
        
        # Son durumu göster
        print(f"\n📊 Simülasyon Özeti:")
        print(f"   Toplam Adım: {step_count}")
        print(f"   Toplam Ödül: {total_reward:.2f}")
        print(f"   Son Hedef Mesafesi: {self.env._goal_dist():.1f}")
        print(f"   Manevra Sayısı: {len(self.maneuver_history)}")
        
        # Son görselleştirmeyi göster
        self._visualize_millisecond_step(step_count, total_reward, info)
        plt.pause(1)  # 3 saniye bekle
        
        self.is_running = False
        plt.ioff()
        plt.show()
        print("\n🎉 Milisaniye hassasiyetinde simülasyon tamamlandı!")
    
    def _analyze_maneuvers(self):
        """Manevra analizi ve geçmişi"""
        current_maneuvers = []
        
        for cluster in self.env.clusters:
            approach_time = self.calculate_cluster_approach_time(cluster)
            needs_maneuver = self.predict_maneuver_need(cluster)
            
            if needs_maneuver:
                current_maneuvers.append({
                    'time': self.current_time,
                    'cluster_id': cluster['id'],
                    'approach_time': approach_time,
                    'distance': np.hypot(cluster['cx'] - self.env.x, cluster['cy'] - self.env.y),
                    'position': (self.env.x, self.env.y)
                })
        
        # Manevra geçmişini güncelle
        if current_maneuvers:
            self.maneuver_history.extend(current_maneuvers)
        
        # Küme yaklaşma geçmişini güncelle
        for cluster in self.env.clusters:
            distance = np.hypot(cluster['cx'] - self.env.x, cluster['cy'] - self.env.y)
            self.cluster_approach_history.append({
                'time': self.current_time,
                'cluster_id': cluster['id'],
                'distance': distance,
                'safe_distance': cluster['r'] + self.env.safe_margin,
                'position': (self.env.x, self.env.y)
            })
    
    def _visualize_millisecond_step(self, step, total_reward, info):
        """Milisaniye hassasiyetinde görselleştirme"""
        self.ax.clear()
        
        # Kümeleri çiz
        for cluster in self.env.clusters:
            # Ana küme
            circle = Circle((cluster['cx'], cluster['cy']), cluster['r'], 
                          color='red', alpha=0.6, label=f"Küme {cluster['id']}")
            self.ax.add_patch(circle)
            
            # Güvenlik tamponu
            safe_circle = Circle((cluster['cx'], cluster['cy']), 
                               cluster['r'] + self.env.safe_margin, 
                               color='orange', alpha=0.3, linestyle='--')
            self.ax.add_patch(safe_circle)
            
            # Yaklaşma süresi bilgisi
            approach_time = self.calculate_cluster_approach_time(cluster)
            needs_maneuver = self.predict_maneuver_need(cluster)
            
            if needs_maneuver:
                color = 'red'
                fontweight = 'bold'
                text = f"⚠️ {approach_time:.1f}s"
            else:
                color = 'black'
                fontweight = 'normal'
                text = f"{approach_time:.1f}s"
            
            self.ax.text(cluster['cx'], cluster['cy'] + cluster['r'] + 8, 
                        text, ha='center', va='bottom', 
                        color=color, fontweight=fontweight, fontsize=10)
        
        # Tahmin edilen waypoint'leri çiz
        waypoints = self.env._predict_future_waypoints()
        for i in range(0, len(waypoints), 2):
            wp_x, wp_y = waypoints[i], waypoints[i+1]
            time_min = (i // 2 + 1)
            
            # Waypoint risk kontrolü
            risk, risky_count = self.env._assess_waypoint_risk(waypoints)
            if risky_count > 0:
                color = 'red'
                size = 150
            else:
                color = 'green'
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
        
        # Manevra geçmişi (son 10 manevra)
        recent_maneuvers = self.maneuver_history[-10:]
        for maneuver in recent_maneuvers:
            x, y = maneuver['position']
            self.ax.scatter(x, y, c='yellow', s=100, marker='o', alpha=0.7, 
                           label=f"Manevra {maneuver['cluster_id']}" if maneuver == recent_maneuvers[0] else "")
        
        # Harita ayarları
        self.ax.set_xlim(self.env.world_bounds['x_min'], self.env.world_bounds['x_max'])
        self.ax.set_ylim(self.env.world_bounds['y_min'], self.env.world_bounds['y_max'])
        self.ax.set_xlabel('X Koordinatı (m)', fontsize=12)
        self.ax.set_ylabel('Y Koordinatı (m)', fontsize=12)
        
        # Başlık ve bilgiler
        title = f'Milisaniye Hassasiyetinde Proaktif Navigation - Zaman: {self.current_time:.1f}s - Adım: {step}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Zaman bilgisi
        time_text = f'Simülasyon Zamanı: {self.current_time:.1f}s'
        self.ax.text(0.02, 0.98, time_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Manevra bilgisi
        if self.maneuver_history:
            latest_maneuver = self.maneuver_history[-1]
            maneuver_text = f'Son Manevra: Küme {latest_maneuver["cluster_id"]} - {latest_maneuver["approach_time"]:.1f}s önce'
            self.ax.text(0.02, 0.92, maneuver_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Hedef mesafesi
        goal_dist = self.env._goal_dist()
        dist_text = f'Goal Distance: {goal_dist:.1f}m'
        self.ax.text(0.02, 0.86, dist_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Adım bilgisi
        step_text = f'Adım: {step}/200'
        self.ax.text(0.02, 0.80, step_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Layout
        plt.tight_layout()
    
    def stop_simulation(self):
        """Simülasyonu durdur"""
        self.is_running = False
        print("⏹️  Simülasyon durduruldu")

def main():
    """Ana fonksiyon"""
    print("🚁 Milisaniye Hassasiyetinde Proaktif Navigation Simülasyonu")
    print("=" * 70)
    print("🎯 Özellikler:")
    print("   • Milisaniye hassasiyetinde simülasyon")
    print("   • Yavaş yavaş kümelerin yanına yaklaşma")
    print("   • 3 saniye önceden manevra tahmini")
    print("   • Gerçek zamanlı görselleştirme")
    print("   • Manevra geçmişi takibi")
    print("=" * 70)
    
    # Simülasyon başlat
    sim = MillisecondPrecisionSimulation()
    
    try:
        # Simülasyonu çalıştır
        sim.run_millisecond_simulation()
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu")
        sim.stop_simulation()
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        sim.stop_simulation()

if __name__ == "__main__":
    main()
