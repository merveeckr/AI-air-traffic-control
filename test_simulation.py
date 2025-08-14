"""
Milisaniye Hassasiyetinde Simülasyon Test Dosyası
Simülasyonun doğru çalıştığını test et
"""

import sys
import os

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Gerekli kütüphanelerin import edilebilir olduğunu test et"""
    print("🧪 Import Testleri Başlıyor...")
    
    try:
        import numpy as np
        print(" numpy import edildi")
    except ImportError as e:
        print(f" numpy import hatası: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(" matplotlib import edildi")
    except ImportError as e:
        print(f" matplotlib import hatası: {e}")
        return False
    
    try:
        from proactive_navigation_env import ProactiveNavigationEnv
        print(" ProactiveNavigationEnv import edildi")
    except ImportError as e:
        print(f" ProactiveNavigationEnv import hatası: {e}")
        return False
    
    try:
        from proactive_training import ProactiveNavigationTrainer
        print(" ProactiveNavigationTrainer import edildi")
    except ImportError as e:
        print(f" ProactiveNavigationTrainer import hatası: {e}")
        return False
    
    return True

def test_environment():
    """Environment'ın çalıştığını test et"""
    print("\n Environment Testi Başlıyor...")
    
    try:
        from proactive_navigation_env import ProactiveNavigationEnv
        
        env = ProactiveNavigationEnv()
        obs, info = env.reset()
        
        print(f" Environment oluşturuldu")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        print(f"   World bounds: {env.world_bounds}")
        
        # Birkaç adım test et
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: Reward = {reward:.2f}, Position = ({env.x:.1f}, {env.y:.1f})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f" Environment test hatası: {e}")
        return False

def test_simulation_class():
    """Simülasyon sınıfının oluşturulabildiğini test et"""
    print("\n🚁 Simülasyon Sınıfı Testi Başlıyor...")
    
    try:
        from millisecond_precision_simulation import MillisecondPrecisionSimulation
        
        sim = MillisecondPrecisionSimulation()
        print(" MillisecondPrecisionSimulation sınıfı oluşturuldu")
        print(f"   dt: {sim.dt}")
        print(f"   Simulation speed: {sim.simulation_speed}")
        print(f"   Visualization FPS: {sim.visualization_fps}")
        
        return True
        
    except Exception as e:
        print(f" Simülasyon sınıfı test hatası: {e}")
        return False

def test_cluster_scenario():
    """Küme senaryosunun oluşturulabildiğini test et"""
    print("\n Küme Senaryosu Testi Başlıyor...")
    
    try:
        from millisecond_precision_simulation import MillisecondPrecisionSimulation
        
        sim = MillisecondPrecisionSimulation()
        obs, info = sim.create_cluster_scenario()
        
        print(" Küme senaryosu oluşturuldu")
        print(f"   Toplam küme sayısı: {len(sim.env.clusters)}")
        
        for cluster in sim.env.clusters:
            print(f"   Küme {cluster['id']}: ({cluster['cx']:.1f}, {cluster['cy']:.1f}) - Risk: {cluster['risk']:.1f}")
        
        return True
        
    except Exception as e:
        print(f" Küme senaryosu test hatası: {e}")
        return False

def run_all_tests():
    """Tüm testleri çalıştır"""
    print(" Milisaniye Hassasiyetinde Simülasyon Testleri Başlıyor...")
    print("=" * 60)
    
    tests = [
        ("Import Testleri", test_imports),
        ("Environment Testi", test_environment),
        ("Simülasyon Sınıfı Testi", test_simulation_class),
        ("Küme Senaryosu Testi", test_cluster_scenario)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f" {test_name} BAŞARILI")
            else:
                print(f" {test_name} BAŞARISIZ")
        except Exception as e:
            print(f" {test_name} HATA: {e}")
    
    print("\n" + "=" * 60)
    print(f" Test Sonuçları: {passed}/{total} BAŞARILI")
    
    if passed == total:
        print(" Tüm testler başarılı! Simülasyon çalıştırılmaya hazır.")
        print("\n Simülasyonu çalıştırmak için:")
        print("   python millisecond_precision_simulation.py")
    else:
        print("  Bazı testler başarısız. Lütfen hataları düzeltin.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
