"""
Hızlı Fine-Tune Scripti
Son eğitilmiş modeli otomatik bulur, yeni ortam mantığıyla 50k–200k adım daha eğitir
ve aynı eğitim klasörüne `final_model.zip` olarak kaydeder.

Kullanım:
  python finetune.py
"""

import os
import sys
from datetime import datetime

from stable_baselines3 import PPO

from proactive_navigation_env import ProactiveNavigationEnv


FINE_TUNE_STEPS = 100_000  
SAFE_MARGIN = 45.0         
NUM_CLUSTERS = 2
GOAL_RADIUS = 40.0
MAX_STEPS = 600


def find_latest_model_path(base_dir: str = "."):
    """`proactive_navigation_training_YYYYMMDD_HHMMSS` klasörleri arasında
    model içeren en güncel klasörü bulur; final > best > checkpoint önceliği uygular.
    """
    training_dirs = [d for d in os.listdir(base_dir)
                     if d.startswith("proactive_navigation_training_")
                     and os.path.isdir(os.path.join(base_dir, d))]
    if not training_dirs:
        raise FileNotFoundError("Eğitim dizini bulunamadı (proactive_navigation_training_*)")

    def parse_timestamp(dir_name: str) -> datetime:
        ts = dir_name.replace("proactive_navigation_training_", "")
        try:
            return datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except Exception:
            return datetime.min

    # Adayları topla: (timestamp, priority, dir, path)
    # priority: final=3, best=2, checkpoint=1
    candidates = []
    for d in training_dirs:
        ts = parse_timestamp(d)
        dir_path = os.path.join(base_dir, d)
        final_model_path = os.path.join(dir_path, "final_model.zip")
        if os.path.exists(final_model_path):
            candidates.append((ts, 3, d, final_model_path))
        best_model_path = os.path.join(dir_path, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            candidates.append((ts, 2, d, best_model_path))
        ckpt_dir = os.path.join(dir_path, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
                latest_ckpt = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                candidates.append((ts, 1, d, os.path.join(ckpt_dir, latest_ckpt)))

    if not candidates:
        # Hata mesajını daha açıklayıcı yap
        missing = "\n".join(sorted(training_dirs))
        raise FileNotFoundError("Hiçbir eğitim klasöründe model bulunamadı. İncelenen klasörler:\n" + missing)

    # En güncel timestamp ve en yüksek önceliğe göre seç
    ts, pr, d, path = max(candidates, key=lambda t: (t[0], t[1]))
    return d, path


def main():
    print("🚀 Fine-tune başlatılıyor...")
    # İsteğe bağlı: komut satırından model dosya yolu verildiyse onu kullan
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        model_path = sys.argv[1]
        # Eğitim klasörü olarak modelin bulunduğu üst klasörü kullan
        train_dir = os.path.basename(os.path.dirname(model_path))
    else:
        train_dir, model_path = find_latest_model_path()
    print(f"📁 Eğitim dizini: {train_dir}")
    print(f"🧠 Yüklenecek model: {model_path}")

    # Ortamı yeni mantıkla kur
    env = ProactiveNavigationEnv()
    env.dynamic_difficulty = False
    env.set_curriculum(
        curriculum_num_clusters=NUM_CLUSTERS,
        goal_radius=GOAL_RADIUS,
        max_steps=MAX_STEPS,
        safe_margin=SAFE_MARGIN,
    )

    # Modeli ortamla birlikte yükle (n_envs uyuşmazlığı uyarısını önlemek için)
    model = PPO.load(model_path, env=env)

    print(f"⏱️  Fine-tune adım sayısı: {FINE_TUNE_STEPS:,}")
    model.learn(total_timesteps=FINE_TUNE_STEPS, reset_num_timesteps=False, progress_bar=True)

    # Aynı eğitim klasörüne final_model.zip olarak kaydet → simülasyon otomatik bulur
    save_path = os.path.join(train_dir, "final_model")
    model.save(save_path)
    print(f"💾 Kaydedildi: {save_path}.zip")
    print("🎉 Fine-tune tamamlandı. Simülasyonu tekrar çalıştırabilirsiniz: python millisecond_precision_simulation.py")


if __name__ == "__main__":
    main()


