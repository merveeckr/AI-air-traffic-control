"""
Proaktif Aircraft Navigation Training Script
Waypoint tahmini + proaktif risk kaçınma için PPO eğitimi
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.utils import set_random_seed
import torch
import os
from datetime import datetime
import json

from proactive_navigation_env import ProactiveNavigationEnv

class ProactiveNavigationTrainer:
    """Proaktif navigasyon eğitim sınıfı"""
    
    def __init__(self):
        """Eğitim konfigürasyonu"""
        self.config = {
            'total_timesteps': 500_000,
            'learning_rate': 1e-4,
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.995,
            'gae_lambda': 0.98,
            'clip_range': 0.2,
            'ent_coef': 0.02,
            'vf_coef': 0.5,
            'max_grad_norm': 0.3,
            'n_envs': 16,
            'eval_freq': 10000,
            'save_freq': 100000,
            'eval_episodes': 20,
            'seed': 42
        }
        
        # Eğitim dizini
        self.training_dir = f"proactive_navigation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.training_dir, exist_ok=True)
        
        print(f"📁 Eğitim dizini: {self.training_dir}")
        
        # Random seed
        set_random_seed(self.config['seed'])
    
    def setup_environment(self):
        """Ortamı kur"""
        print("🔧 Proaktif Navigation Ortamı kuruluyor...")
        
        # Vectorized environment
        self.env = make_vec_env(
            lambda: Monitor(ProactiveNavigationEnv()),
            n_envs=self.config['n_envs'],
            seed=self.config['seed']
        )
        
        # Değerlendirme ortamı
        self.eval_env = make_vec_env(
            lambda: Monitor(ProactiveNavigationEnv()),
            n_envs=1,
            seed=self.config['seed'] + 1000
        )
        
        print(f"✅ {self.config['n_envs']} paralel ortam kuruldu")
        print(f"📊 Observation Space: {self.env.observation_space}")
        print(f"🎯 Action Space: {self.env.action_space}")
    
    def setup_model(self):
        """PPO modelini kur"""
        print("🧠 Proaktif Navigation PPO Model kuruluyor...")
        
        # Policy network - Yüksek başarı oranı için optimize edilmiş
        policy_kwargs = {
            'net_arch': [
                dict(
                    pi=[1024, 1024, 512, 256, 128],  # Daha da derin policy network
                    vf=[1024, 1024, 512, 256, 128]   # Daha da derin value network
                )
            ],
            'activation_fn': torch.nn.ReLU,
            'ortho_init': True
        }
        
        # PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"{self.training_dir}/tensorboard_logs"
        )
        
        print("✅ Proaktif Navigation PPO Model kuruldu")
        print(f"📊 Model parametreleri: {sum(p.numel() for p in self.model.policy.parameters()):,}")
    
    def setup_callbacks(self):
        """Callback'leri kur"""
        print("📞 Callback'ler kuruluyor...")
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['save_freq'],
            save_path=f"{self.training_dir}/checkpoints",
            name_prefix="proactive_navigation_ppo"
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.training_dir}/best_model",
            log_path=f"{self.training_dir}/eval_logs",
            eval_freq=self.config['eval_freq'],
            n_eval_episodes=self.config['eval_episodes'],
            deterministic=True,
            render=False
        )
        
        self.callbacks = [checkpoint_callback, eval_callback]
        print("✅ Callback'ler kuruldu")
    
    def train(self):
        """Modeli eğit - curriculum ile"""
        print("🚀 Proaktif Navigation PPO Eğitimi (curriculum) başlatılıyor...")
        print(f"⏱️  Toplam adım: {self.config['total_timesteps']:,}")
        try:
            # Phase 1: kolay (1 cluster)
            print("📘 Curriculum Phase 1: 1 cluster, goal_radius=60, max_steps=300")
            self.env.env_method('set_curriculum', goal_radius=60, max_steps=300, curriculum_num_clusters=1)
            self.model.learn(total_timesteps=100_000, callback=self.callbacks, progress_bar=True)
            # Phase 2: orta (2 cluster)
            print("📙 Curriculum Phase 2: 2 cluster, goal_radius=45, max_steps=400")
            self.env.env_method('set_curriculum', goal_radius=45, max_steps=400, curriculum_num_clusters=2)
            self.model.learn(total_timesteps=200_000, callback=self.callbacks, progress_bar=True)
            # Phase 3: zor (2-3 cluster)
            print("📕 Curriculum Phase 3: 2-3 cluster, goal_radius=40, max_steps=500")
            self.env.env_method('set_curriculum', goal_radius=40, max_steps=500, curriculum_num_clusters=0)
            remaining = max(0, self.config['total_timesteps'] - 300_000)
            if remaining > 0:
                self.model.learn(total_timesteps=remaining, callback=self.callbacks, progress_bar=True)
            # Final modeli kaydet
            final_model_path = f"{self.training_dir}/final_model"
            self.model.save(final_model_path)
            print(f"💾 Final model kaydedildi: {final_model_path}")
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")
            raise
    
    def evaluate_model(self, num_episodes=20):
        """Eğitilmiş modeli değerlendir"""
        print(f"🔍 Proaktif Navigation Model değerlendiriliyor ({num_episodes} episode)...")
        
        if self.model is None:
            print("❌ Model bulunamadı!")
            return
        
        # Değerlendirme ortamı
        eval_env = ProactiveNavigationEnv()
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        collision_count = 0
        out_of_bounds_count = 0
        proactive_avoidance_count = 0
        timeout_count = 0
        
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            step_count = 0
            episode_success = False
            
            while True:
                # Model action'ı
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # Başarı durumunu takip et
                if info.get('success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # İstatistikler - Yeni success tracking ile
            if episode_success:
                success_count += 1
            
            if info.get('collision', False):
                collision_count += 1
            
            if info.get('out_of_bounds', False):
                out_of_bounds_count += 1
            
            if info.get('episode_type') == 'timeout':
                timeout_count += 1
            
            # Proaktif kaçınma analizi
            waypoints = eval_env._predict_future_waypoints()
            risk, risky_count = eval_env._assess_waypoint_risk(waypoints)
            if risky_count == 0:
                proactive_avoidance_count += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
            print(f"  Success: {episode_success}, Episode Type: {info.get('episode_type', 'unknown')}")
            print(f"  Waypoint Risk: {risk:.2f}, Risky Clusters: {risky_count}")
        
        eval_env.close()
        
        # Sonuçları yazdır
        print("\n📊 Proaktif Navigation Değerlendirme Sonuçları:")
        print(f"   Başarı Oranı: {success_count/num_episodes*100:.1f}%")
        print(f"   Çarpışma Oranı: {collision_count/num_episodes*100:.1f}%")
        print(f"   Sınır Dışı Oranı: {out_of_bounds_count/num_episodes*100:.1f}%")
        print(f"   Timeout Oranı: {timeout_count/num_episodes*100:.1f}%")
        print(f"   Proaktif Kaçınma Oranı: {proactive_avoidance_count/num_episodes*100:.1f}%")
        print(f"   Ortalama Ödül: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Ortalama Uzunluk: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        return {
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'out_of_bounds_rate': out_of_bounds_count / num_episodes,
            'timeout_rate': timeout_count / num_episodes,
            'proactive_avoidance_rate': proactive_avoidance_count / num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }
    
    def save_training_config(self):
        """Eğitim konfigürasyonunu kaydet"""
        config_path = f"{self.training_dir}/training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"📋 Konfigürasyon kaydedildi: {config_path}")
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        # En son checkpoint'i bul
        checkpoint_dir = f"{self.training_dir}/checkpoints"
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError("Checkpoint dizini bulunamadı")
        
        # En son checkpoint dosyasını bul
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
        if not checkpoint_files:
            raise FileNotFoundError("Checkpoint dosyası bulunamadı")
        
        # En son checkpoint'i seç (en büyük step numarası)
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"📥 Model yükleniyor: {checkpoint_path}")
        model = PPO.load(checkpoint_path)
        print("✅ Model başarıyla yüklendi!")
        
        return model
    
    def train_model(self):
        """Modeli eğit ve döndür"""
        print("🚀 Model eğitimi başlatılıyor...")
        
        # Ortamı kur
        self.setup_environment()
        
        # Modeli kur
        self.setup_model()
        
        # Callback'leri kur
        self.setup_callbacks()
        
        # Konfigürasyonu kaydet
        self.save_training_config()
        
        # Eğitimi başlat
        self.train()
        
        print("✅ Model eğitimi tamamlandı!")
        return self.model
    
    def plot_training_results(self):
        """Eğitim sonuçlarını görselleştir"""
        print("📈 Eğitim sonuçları görselleştiriliyor...")
        
        try:
            log_dir = f"{self.training_dir}/tensorboard_logs"
            if os.path.exists(log_dir):
                plot_results([log_dir], self.config['total_timesteps'], 'timesteps')
                plt.savefig(f"{self.training_dir}/training_results.png", dpi=300, bbox_inches='tight')
                print(f"📊 Eğitim grafiği kaydedildi: {self.training_dir}/training_results.png")
                plt.show()
            else:
                print("⚠️  Tensorboard log'ları bulunamadı")
        except Exception as e:
            print(f"⚠️  Grafik oluşturma hatası: {e}")
    
    def run_complete_training(self):
        """Tam eğitim sürecini çalıştır"""
        print("🚀 Proaktif Aircraft Navigation Eğitim Süreci Başlatılıyor...")
        print("=" * 70)
        
        try:
            # 1. Ortamı kur
            self.setup_environment()
            
            # 2. Modeli kur
            self.setup_model()
            
            # 3. Callback'leri kur
            self.setup_callbacks()
            
            # 4. Konfigürasyonu kaydet
            self.save_training_config()
            
            # 5. Eğitimi başlat
            self.train()
            
            # 6. Sonuçları görselleştir
            self.plot_training_results()
            
            # 7. Modeli değerlendir
            print("\n" + "=" * 70)
            self.evaluate_model(num_episodes=20)
            
            print("\n🎉 Proaktif Navigation eğitim süreci başarıyla tamamlandı!")
            
        except Exception as e:
            print(f"\n❌ Eğitim sürecinde hata: {e}")
            raise
        finally:
            # Ortamları kapat
            if hasattr(self, 'env') and self.env:
                self.env.close()
            if hasattr(self, 'eval_env') and self.eval_env:
                self.eval_env.close()

def main():
    """Ana fonksiyon"""
    print("🚀 Proaktif Aircraft Navigation Training System")
    print("=" * 70)
    
    # Eğitimi başlat
    trainer = ProactiveNavigationTrainer()
    trainer.run_complete_training()

if __name__ == "__main__":
    main()
