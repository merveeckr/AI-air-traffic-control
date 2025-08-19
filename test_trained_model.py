"""
Eğitilmiş Proaktif Navigation Modelini Test Et
Zaten eğitilmiş modeli yükleyip performansını test eder
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
from proactive_navigation_env import ProactiveNavigationEnv

def load_and_test_model(model_path):
    """Eğitilmiş modeli yükle ve test et"""
    print(f"🧠 Model yükleniyor: {model_path}")
    
    try:
        # Modeli yükle
        model = PPO.load(model_path)
        print("✅ Model başarıyla yüklendi!")
        
        # Ortamı oluştur
        env = ProactiveNavigationEnv(render_mode="human")
        
        # Test episode'u çalıştır
        print("🚀 Test episode'u başlatılıyor...")
        
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        while step_count < 1000:  # Maksimum 1000 adım
            # Modelden action al
            action, _states = model.predict(obs, deterministic=True)
            
            # Ortamı adım adım ilerlet
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Her 50 adımda bilgi ver
            if step_count % 50 == 0:
                print(f"Adım {step_count}: Ödül = {reward:.2f}, Toplam = {total_reward:.2f}")
                print(f"Pozisyon: ({env.x:.1f}, {env.y:.1f})")
                print(f"Hedef Mesafesi: {env._goal_dist():.1f}")
            
            # Episode bitti mi kontrol et
            if terminated or truncated:
                print(f"Episode {step_count} adımda tamamlandı")
                break
        
        print(f"🎯 Final Toplam Ödül: {total_reward:.2f}")
        print(f"📊 Ortalama Adım Ödülü: {total_reward/step_count:.2f}")
        
        # Görselleştir
        env.render()
        plt.show()
        
        env.close()
        return model, total_reward
        
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return None, 0

def list_available_models():
    """Mevcut eğitilmiş modelleri listele"""
    print("📁 Mevcut eğitilmiş modeller:")
    
    training_dirs = [d for d in os.listdir('.') if d.startswith('proactive_navigation_training_')]
    
    for dir_name in training_dirs:
        print(f"\n📂 {dir_name}:")
        
        # Best model
        best_model_path = os.path.join(dir_name, 'best_model', 'best_model.zip')
        if os.path.exists(best_model_path):
            print(f"  🏆 Best Model: {best_model_path}")
        
        # Final model
        final_model_path = os.path.join(dir_name, 'final_model.zip')
        if os.path.exists(final_model_path):
            print(f"  🎯 Final Model: {final_model_path}")
        
        # Checkpoints
        checkpoint_dir = os.path.join(dir_name, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
            if checkpoints:
                print(f"  💾 Checkpoints: {len(checkpoints)} adet")
                for cp in checkpoints[:3]:  # İlk 3'ünü göster
                    print(f"    - {cp}")
                if len(checkpoints) > 3:
                    print(f"    ... ve {len(checkpoints)-3} tane daha")

def main():
    """Ana fonksiyon"""
    print("🤖 Eğitilmiş Proaktif Navigation Model Testi")
    print("=" * 50)
    
    # Mevcut modelleri listele
    list_available_models()
    
    print("\n" + "=" * 50)
    
    # Kullanıcıdan model seçmesini iste
    print("\n🎯 Test etmek istediğiniz modeli seçin:")
    print("1. En son eğitilen best model")
    print("2. En son eğitilen final model")
    print("3. Manuel model yolu girin")
    
    choice = input("Seçiminiz (1-3): ").strip()
    
    if choice == "1":
        # En son eğitilen best model
        training_dirs = sorted([d for d in os.listdir('.') if d.startswith('proactive_navigation_training_')])
        if training_dirs:
            latest_dir = training_dirs[-1]
            model_path = os.path.join(latest_dir, 'best_model', 'best_model.zip')
            if os.path.exists(model_path):
                print(f"🎯 Seçilen model: {model_path}")
                load_and_test_model(model_path)
            else:
                print("❌ Best model bulunamadı")
        else:
            print("❌ Eğitim dizini bulunamadı")
    
    elif choice == "2":
        # En son eğitilen final model
        training_dirs = sorted([d for d in os.listdir('.') if d.startswith('proactive_navigation_training_')])
        if training_dirs:
            latest_dir = training_dirs[-1]
            model_path = os.path.join(latest_dir, 'final_model.zip')
            if os.path.exists(model_path):
                print(f"🎯 Seçilen model: {model_path}")
                load_and_test_model(model_path)
            else:
                print("❌ Final model bulunamadı")
        else:
            print("❌ Eğitim dizini bulunamadı")
    
    elif choice == "3":
        # Manuel model yolu
        model_path = input("Model dosya yolu: ").strip()
        if os.path.exists(model_path):
            load_and_test_model(model_path)
        else:
            print("❌ Belirtilen dosya bulunamadı")
    
    else:
        print("❌ Geçersiz seçim")

if __name__ == "__main__":
    main()
