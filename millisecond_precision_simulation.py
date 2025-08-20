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
        self.dt = 0.2  # 0.2 saniye (daha görünür)
        self.simulation_speed = 2  # 0.1x hız (çok daha yavaş)
        self.visualization_fps = 5   # 5 FPS (çok daha yavaş görselleştirme)
        
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
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Mevcut dizin
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
        
        # En son geçerli dizini seç - Tarih bazlı sıralama
        # Dizin adlarındaki tarih formatını parse et: proactive_navigation_training_YYYYMMDD_HHMMSS
        def parse_timestamp(dir_name):
            timestamp = dir_name.replace('proactive_navigation_training_', '')
        # YYYYMMDD_HHMMSS formatını datetime'a çevir
            try:
                return datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            except:
                return datetime.min
    
        # En son eğitimi bul
        latest_dir = max(training_dirs, key=parse_timestamp)
        print(f"🔍 En son eğitim: {latest_dir}")
        
        # En son eğitilen modeli bul - latest_dir'ı kullan
        print(f"🔍 Aranan latest_dir: {latest_dir}")
        print(f"🔍 Mevcut valid_dirs:")
        for vd in valid_dirs:
            print(f"   - {vd[0]} (model: {vd[1]})")
        
        latest_valid_dir = None
        for valid_dir in valid_dirs:
            print(f"🔍 Kontrol: {valid_dir[0]} == {latest_dir} ? {valid_dir[0] == latest_dir}")
            if valid_dir[0] == latest_dir:
                latest_valid_dir = valid_dir
                print(f"✅ Eşleşme bulundu: {valid_dir}")
                break
        
        if latest_valid_dir is None:
            print(f"⚠️  latest_dir'da model bulunamadı, fallback kullanılıyor")
            # Eğer latest_dir'da model yoksa, en son valid_dir'ı kullan
            latest_valid_dir = max(valid_dirs, key=lambda x: parse_timestamp(x[0]))
        
        dir_name, model_file, model_path = latest_valid_dir
        
        print(f"🔍 Tarih sıralaması:")
        for valid_dir in sorted(valid_dirs, key=lambda x: parse_timestamp(x[0])):
            print(f"   {parse_timestamp(valid_dir[0])} -> {valid_dir[0]}")
        print(f"   En son seçilen: {parse_timestamp(dir_name)} -> {dir_name}")
        
        print(f"📁 Model bulundu: {dir_name}")
        print(f"📥 Model dosyası: {model_file}")
        print(f"📂 Tam yol: {model_path}")
        
        return PPO.load(model_path)
    
    def create_cluster_scenario(self):
        """Cluster'ın arkasında hedef ile manevra senaryosu"""
        print("🎯 Cluster Arkasında Hedef + Manevra Senaryosu Oluşturuluyor...")

        # Kontrollü senaryo için dinamik zorluğu kapat ve reset et
        self.env.dynamic_difficulty = False
        obs, info = self.env.reset()

        # Environment parametrelerini ayarla - Optimize edilmiş
        self.env.max_steps = 1000  # Uzun uçuş
        self.env.goal_radius = 25.0   # Büyük hedef (15.0'dan 25.0'a) - daha kolay başarı
        self.env.safe_margin = 45.0  # Küçük güvenlik mesafesi (45.0'dan 30.0'a) - daha az dönüş
        
        # TEK CLUSTER TEST - Basit senaryo
        self.env.clusters = []
        
        # Tek küme: Yol üzerinde ama geçilebilir
        cluster1 = {
            'cx': 35.0,  # Yol üzerinde ama daha uzağa
            'cy': 25.0,
            'r': 3.0,    # Makul boyut
            'risk': 0.4,  # Düşük risk
            'id': 1
        }
        
        self.env.clusters = [cluster1]  # SADECE 1 CLUSTER!
        
        # Hedefi cluster'ın arkasına yerleştir - Basit hesaplama
        cluster = self.env.clusters[0]  # Tek cluster
        angle = np.pi / 4  # 45 derece
        distance = cluster['r'] + 25  # Güvenli mesafe
        
        self.env.goal = np.array([
            cluster['cx'] + distance * np.cos(angle),
            cluster['cy'] + distance * np.sin(angle)
        ])
        
        print("✅ Cluster arkasında hedef + manevra senaryosu oluşturuldu:")
        print(f"   Environment Max Steps: {self.env.max_steps}")
        print(f"   Environment Goal Radius: {self.env.goal_radius}")
        print(f"   Environment Safe Margin: {self.env.safe_margin}")
        print(f"   Hedef: ({self.env.goal[0]:.1f}, {self.env.goal[1]:.1f})")
        for cluster in self.env.clusters:
            print(f"   Küme {cluster['id']}: ({cluster['cx']:.1f}, {cluster['cy']:.1f}) - Risk: {cluster['risk']:.1f} - Yarıçap: {cluster['r']:.1f}")

        # Güncellenmiş state'i döndür
        obs = self.env._get_obs()
        info = {"scenario": "single_cluster"}
        return obs, info
    
    def create_two_cluster_scenario(self):
        """İkili cluster senaryosu kur (iki manevrayı test etmek için)"""
        # Ortam parametreleri
        self.env.max_steps = 600
        self.env.goal_radius = 40.0
        self.env.safe_margin = 45.0
        # Rastgele zorlukları kapat ve 2 cluster zorla
        self.env.dynamic_difficulty = False
        self.env.set_curriculum(curriculum_num_clusters=2, goal_radius=self.env.goal_radius, max_steps=self.env.max_steps, safe_margin=self.env.safe_margin)
        # Reset ve kümeleri manuel yerleştir
        self.env.reset()
        self.env.clusters = [
            {'cx': 30.0, 'cy': 20.0, 'r': 6.0, 'risk': 0.5, 'id': 1},
            {'cx': 55.0, 'cy': 35.0, 'r': 6.0, 'risk': 0.5, 'id': 2},
        ]
        # Hedefi ikinci cluster’ın arkasına yerleştir
        angle = np.pi/3
        dist = 25.0
        self.env.goal = np.array([
            self.env.clusters[1]['cx'] + dist*np.cos(angle),
            self.env.clusters[1]['cy'] + dist*np.sin(angle)
        ])
        print("✅ 2 Cluster senaryosu hazır:")
        for c in self.env.clusters:
            print(f"  Küme {c['id']}: ({c['cx']}, {c['cy']}) r={c['r']}")
        print(f"  Hedef: ({self.env.goal[0]:.1f}, {self.env.goal[1]:.1f})")
    
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
        """1.5 saniye önceden manevra ihtiyacını tahmin et"""
        approach_time = self.calculate_cluster_approach_time(cluster)
        return approach_time <= 1.5  # 1.5 saniye veya daha az - daha erken manevra
    
    def run_millisecond_simulation(self):
        """Milisaniye hassasiyetinde simülasyon"""
        print("🚀 Milisaniye Hassasiyetinde Simülasyon Başlıyor...")
        print(f"⏱️  Simülasyon Hızı: {self.simulation_speed}x")
        print(f"📊 Görselleştirme: {self.visualization_fps} FPS")
        
        self.is_running = True
        self.episode_start_time = time.time()
        self.current_time = 0.0
        
        # obs, info = self.create_cluster_scenario()  # Environment zaten reset edildi
        total_reward = 0
        step_count = 0
        
        # Simülasyon döngüsü - Daha uzun süre çalışsın
        max_steps = 500  # En az 500 adım çalışsın (200'den 500'e)
        
        # Environment'ı manuel olarak kontrol et
        self.env.max_steps = max_steps  # Maksimum adım sayısını artır
        self.env.goal_radius = 25.0     # Hedef yarıçapını büyüt (15.0'dan 25.0'a) - daha kolay başarı
        self.env.safe_margin = 45.0     # Daha geniş güvenlik tamponu
        
        # Environment parametrelerini doğrudan override et
        print(f"🔧 Environment parametreleri override ediliyor...")
        print(f"   Önceki Safe Margin: {getattr(self.env, 'safe_margin', 'Bilinmiyor')}")
        self.env.safe_margin = 45.0
        print(f"   Yeni Safe Margin: {self.env.safe_margin}")
        
        # Environment'ı reset et ve başlangıç durumunu ayarla
        obs, info = self.env.reset()
        self.env.safe_margin = 45.0  # Reset'ten sonra sabit tut
        
        # Cluster'ları tekrar set et (reset'te kaybolmuş olabilir)
        # İki cluster senaryosunu koru
        self.env.clusters = [
            {'cx': 30.0, 'cy': 20.0, 'r': 6.0, 'risk': 0.5, 'id': 1},
            {'cx': 55.0, 'cy': 35.0, 'r': 6.0, 'risk': 0.5, 'id': 2},
        ]
        # Hedefi ikinci cluster'ın arkasına yerleştir
        angle = np.pi/3
        dist = 25.0
        self.env.goal = np.array([
            self.env.clusters[1]['cx'] + dist*np.cos(angle),
            self.env.clusters[1]['cy'] + dist*np.sin(angle)
        ])
        
        print(f"🔄 Environment reset edildi")
        print(f"   Yeni pozisyon: ({self.env.x:.1f}, {self.env.y:.1f})")
        print(f"   Yeni hedef mesafesi: {self.env._goal_dist():.1f}")
        print(f"   Güncel Safe Margin: {self.env.safe_margin}")
        print(f"   Cluster sayısı: {len(self.env.clusters)}")
        for i, cluster in enumerate(self.env.clusters):
            print(f"   Cluster {cluster['id']}: ({cluster['cx']:.1f}, {cluster['cy']:.1f}) - r: {cluster['r']:.1f} - Risk: {cluster['risk']:.1f}")
        print(f"   Hedef: ({self.env.goal[0]:.1f}, {self.env.goal[1]:.1f})")
        
        # Başlangıç reward'ını 0 olarak ayarla
        reward = 0.0
        
        print(f"🔧 Environment ayarları:")
        print(f"   Max Steps: {self.env.max_steps}")
        print(f"   Goal Radius: {self.env.goal_radius}")
        print(f"   Safe Margin: {self.env.safe_margin}")
        
        print(f"🧠 Model bilgileri:")
        print(f"   Model tipi: {type(self.model)}")
        print(f"   Model yüklendi mi: {self.model is not None}")
        
        print(f"🎯 Başlangıç durumu:")
        print(f"   Pozisyon: ({self.env.x:.1f}, {self.env.y:.1f})")
        print(f"   Hız: {self.env.speed:.1f}")
        print(f"   Yön: {np.rad2deg(self.env.heading):.1f}°")
        print(f"   Hedef mesafesi: {self.env._goal_dist():.1f}")
        
        while self.is_running and step_count < max_steps:
            start_step_time = time.time()
            
            try:
                # AI action'ı al
                action, _ = self.model.predict(obs, deterministic=True)

                # Manevra asistanı: Engel varsa daha kararlı dönüş uygula
                try:
                    if hasattr(self.env, '_is_maneuver_needed') and self.env._is_maneuver_needed():
                        mx, my = self.env._get_maneuver_target()
                        desired = np.arctan2(my - self.env.y, mx - self.env.x)
                        bearing_err = desired - self.env.heading
                        while bearing_err > np.pi: bearing_err -= 2*np.pi
                        while bearing_err < -np.pi: bearing_err += 2*np.pi
                        max_delta = self.env.max_turn_rate * self.env.dt
                        # 1.5x agresif dönüş
                        turn_norm = float(np.clip(1.5 * (bearing_err / max_delta), -1.0, 1.0))
                        if isinstance(action, np.ndarray) and action.shape[0] >= 2:
                            action[0] = turn_norm
                            # dönüşte yavaşla
                            action[1] = min(float(action[1]), -0.3)
                except Exception:
                    pass
                
                # Debug bilgileri
                if step_count % 5 == 0:  # Her 5 adımda bir
                    print(f"Adım {step_count}: Action={action}")
                    print(f"   Pozisyon: ({self.env.x:.1f}, {self.env.y:.1f})")
                    print(f"   Hız: {self.env.speed:.1f}")
                    print(f"   Yön: {np.rad2deg(self.env.heading):.1f}°")
                    print(f"   Hedef Mesafesi: {self.env._goal_dist():.1f}")
                    print(f"   Reward: {reward:.3f}")
                    print("   ---")
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                step_count += 1
                self.current_time += self.dt
                
                # Manevra analizi
                self._analyze_maneuvers()
                
                # Her adımda görselleştir (daha sık)
                if step_count % 1 == 0:  # Her adımda görselleştir
                    self._visualize_millisecond_step(step_count, total_reward, info)
                    plt.pause(0.5)  # 0.5 saniye bekle (daha yavaş)
                
                # Episode durumu kontrol et - Daha esnek
                if terminated:
                    print(f"\n🎯 Episode Terminated!")
                    print(f"   Steps: {step_count}")
                    print(f"   Total Reward: {total_reward:.2f}")
                    print(f"   Success: {info.get('success', False)}")
                    print(f"   Final Goal Distance: {self.env._goal_dist():.1f}")
                    print(f"   Reason: {info.get('episode_type', 'unknown')}")

                    # Eğer LOS-success ise ve hedef yarıçapı içinde değilsek, episode'u başarı sayma
                    if info.get('episode_type') == 'los_success' and self.env._goal_dist() >= self.env.goal_radius:
                        print("   ℹ️ LOS-success tetiklendi ama hedef yarıçapı içinde değiliz; devam ediliyor...")
                        terminated = False
                        continue
                    # Eğer timeout near goal ise ve mesafe çok düşükse, kısa bir ek pencere ver
                    if info.get('episode_type') == 'timeout_near_goal' and self.env._goal_dist() > self.env.goal_radius:
                        print("   ⏳ Hedefe çok yakın zaman aşımı; 80 ek adım veriliyor...")
                        extra_steps = 80
                        for _ in range(extra_steps):
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, reward, terminated2, truncated2, info2 = self.env.step(action)
                            total_reward += reward
                            step_count += 1
                            if terminated2 or truncated2:
                                break
                        continue
                    
                    # Başarılı olursa simülasyonu durdur
                    if info.get('success', False):
                        print("   🎉 Hedefe ulaşıldı! Simülasyon başarıyla tamamlandı.")
                        break
                    
                    # Terminated olsa bile devam et (sadece çarpışma durumunda)
                    if info.get('episode_type') == 'collision':
                        print("   ⚠️  Çarpışma tespit edildi, simülasyon durduruluyor.")
                        break
                    elif step_count < 100:  # Çok erken terminate olursa devam et
                        print("   ⚠️  Çok erken terminate, devam ediliyor...")
                        obs, info = self.env.reset()
                        continue
                    else:
                        print("   ⏰ Episode tamamlandı, simülasyon devam ediyor...")
                        obs, info = self.env.reset()
                        continue
                
                if truncated:
                    print(f"\n⏰ Episode Truncated!")
                    print(f"   Steps: {step_count}")
                    print(f"   Total Reward: {total_reward:.2f}")
                    print(f"   Final Goal Distance: {self.env._goal_dist():.1f}")
                    
                    # Hedef mesafesini kontrol et
                    goal_dist = self.env._goal_dist()
                    if goal_dist < self.env.goal_radius * 1.2:  # %20 tolerans
                        print(f"   🎯 Hedef yakınında ({goal_dist:.1f}m), başarılı sayılıyor!")
                        break
                    elif step_count < 100:  # Çok erken truncate olursa devam et
                        print("   ⚠️  Çok erken truncate, devam ediliyor...")
                        obs, info = self.env.reset()
                        continue
                    else:
                        print("   ⏰ Episode tamamlandı, simülasyon devam ediyor...")
                        obs, info = self.env.reset()
                        continue
                
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
        """Manevra analizi ve geçmişi - Waypoint bazlı manevra planlama"""
        current_maneuvers = []
        
        # Waypoint'leri tahmin et
        waypoints = self.env._predict_future_waypoints()
        
        for cluster in self.env.clusters:
            approach_time = self.calculate_cluster_approach_time(cluster)
            needs_maneuver = self.predict_maneuver_need(cluster)
            
            # Waypoint bazlı ek manevra kontrolü
            waypoint_risk = self._check_waypoint_cluster_intersection(waypoints, cluster)
            
            if needs_maneuver or waypoint_risk:
                current_maneuvers.append({
                    'time': self.current_time,
                    'cluster_id': cluster['id'],
                    'approach_time': approach_time,
                    'distance': np.hypot(cluster['cx'] - self.env.x, cluster['cy'] - self.env.y),
                    'position': (self.env.x, self.env.y),
                    'waypoint_risk': waypoint_risk
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
    
    def _check_waypoint_cluster_intersection(self, waypoints, cluster):
        """Waypoint'lerin cluster ile kesişimini kontrol et"""
        for i in range(0, len(waypoints), 2):
            wp_x, wp_y = waypoints[i], waypoints[i+1]
            
            # Waypoint'in cluster'a olan mesafesi
            d = np.hypot(cluster['cx'] - wp_x, cluster['cy'] - wp_y)
            
            # Eğer waypoint güvenlik mesafesi içindeyse risk var
            if d < (cluster['r'] + self.env.safe_margin):
                return True
        
        return False
    
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
        
        # Manevra hedefi göster
        if hasattr(self.env, '_is_maneuver_needed') and self.env._is_maneuver_needed():
            maneuver_target = self.env._get_maneuver_target()
            if maneuver_target != (self.env.goal[0], self.env.goal[1]):
                self.ax.scatter(maneuver_target[0], maneuver_target[1], c='orange', s=200, marker='s', 
                               label='Manevra Hedefi', alpha=0.8, zorder=4)
                
                # Manevra rotası çiz
                self.ax.plot([self.env.x, maneuver_target[0]], [self.env.y, maneuver_target[1]], 
                           'orange', linestyle='--', linewidth=2, alpha=0.6)
        
        # Engel tespiti göster
        if hasattr(self.env, '_detect_cluster_obstacles'):
            obstacles = self.env._detect_cluster_obstacles()
            for obstacle in obstacles:
                # Engel yolunu kırmızı çizgi ile göster (mevcut pozisyondan hedefe)
                self.ax.plot([self.env.x, self.env.goal[0]], [self.env.y, self.env.goal[1]], 
                           'red', linestyle=':', linewidth=3, alpha=0.4, label='Engel Yolu')
                break  # Sadece bir kez çiz
        
        # Harita ayarları
        self.ax.set_xlim(self.env.world_bounds['x_min'], self.env.world_bounds['x_max'])
        self.ax.set_ylim(self.env.world_bounds['y_min'], self.env.world_bounds['y_max'])
        self.ax.set_xlabel('X Koordinatı (m)', fontsize=12)
        self.ax.set_ylabel('Y Koordinatı (m)', fontsize=12)
        
        # Başlık ve bilgiler
        title = f'Cluster Arkasında Hedef + Manevra Simülasyonu - Zaman: {self.current_time:.1f}s - Adım: {step}'
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
        
        # Manevra durumu
        if hasattr(self.env, '_is_maneuver_needed'):
            if self.env._is_maneuver_needed():
                maneuver_status = "Manevra Gerekli"
                maneuver_color = "orange"
                if hasattr(self.env, '_get_maneuver_target'):
                    maneuver_target = self.env._get_maneuver_target()
                    if maneuver_target != (self.env.goal[0], self.env.goal[1]):
                        target_text = f'Manevra Hedefi: ({maneuver_target[0]:.1f}, {maneuver_target[1]:.1f})'
                        self.ax.text(0.02, 0.78, target_text, transform=self.ax.transAxes, 
                                    fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
            else:
                maneuver_status = "Direkt Hedef"
                maneuver_color = "lightgreen"
            
            status_text = f'Durum: {maneuver_status}'
            self.ax.text(0.02, 0.74, status_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=maneuver_color, alpha=0.7))
        
        # Hedef mesafesi
        goal_dist = self.env._goal_dist()
        dist_text = f'Goal Distance: {goal_dist:.1f}m'
        self.ax.text(0.02, 0.86, dist_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Adım bilgisi
        step_text = f'Adım: {step}/200'
        self.ax.text(0.02, 0.68, step_text, transform=self.ax.transAxes, 
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
    print("🚁 Cluster Arkasında Hedef + Manevra Simülasyonu")
    print("=" * 70)
    print("🎯 Özellikler:")
    print("   • Hedef cluster'ın arkasında yerleştirildi")
    print("   • Cluster tespiti ve manevra planlama")
    print("   • Güvenli rota hesaplama")
    print("   • Manevra ödüllendirme sistemi")
    print("   • Gerçek zamanlı görselleştirme")
    print("   • Manevra geçmişi takibi")
    print("=" * 70)
    
    # Simülasyon başlat
    sim = MillisecondPrecisionSimulation()
    
    # İkili cluster senaryosunu zorla (iki manevra testi)
    try:
        sim.create_two_cluster_scenario()
    except Exception as e:
        print(f"⚠️  2 cluster senaryosu kurulamadı: {e}")
    
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
