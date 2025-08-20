

##  Özellikler

- **Proaktif Risk Kaçınma**: 3 saniye önceden waypoint tahmini
- **Dinamik Zorluk**: Başarıya göre artan zorluk seviyesi
- **Gerçek Zamanlı Simülasyon**: Canlı görselleştirme ve kontrol
- **PPO Reinforcement Learning**: Stable-Baselines3 ile eğitim
- **Çoklu Senaryo Testi**: Farklı engel konfigürasyonları

##  Proje Yapısı

```
Yeni klasör/
├── proactive_navigation_env.py      # Ana simülasyon ortamı
├── scenario_sweep.py                # Senaryo test ve görselleştirme
├── proactive_training.py            # Model eğitim sınıfı
├── test_trained_model.py            # Model test
├── millisecond_precision_simulation.py  # Hassas simülasyon
├── data/                            # Eğitim verileri
│   └── proactive_navigation_training_YYYYMMDD_HHMMSS/
│       ├── best_model/              # En iyi model
│       ├── checkpoints/             # Eğitim kontrol noktaları
│       ├── final_model.zip          # Final model
│       └── training_config.json     # Eğitim konfigürasyonu
└── README.md                        # Bu dosya
```

##  Hızlı Başlangıç

### Gereksinimler

```bash
pip install stable-baselines3 gymnasium matplotlib numpy
```

### Temel Kullanım

```bash
# En son eğitilen modeli otomatik bul ve test et
python scenario_sweep.py --render --clusters 3 --safe 40 --goalR 6 --seed 2 --assist

# Belirli bir modeli test et
python scenario_sweep.py ".\data\proactive_navigation_training_20250820_141111\final_model.zip" --render --clusters 3 --safe 40 --goalR 6 --seed 2 --assist
```

## Scenario Sweep Kullanımı

`scenario_sweep.py` dosyası, eğitilmiş modelleri test etmek ve farklı senaryolarda performansını değerlendirmek için kullanılır.

### Komut Satırı Parametreleri

| Parametre | Açıklama | Varsayılan | Örnek |
|-----------|----------|------------|-------|
| `--render` | Görsel simülasyon modu | False | `--render` |
| `--clusters` | Engel kümesi sayısı | 2 | `--clusters 3` |
| `--safe` | Güvenli mesafe (birim) | 40.0 | `--safe 40` |
| `--goalR` | Hedef yarıçapı (birim) | 5.0 | `--goalR 1` |
| `--seed` | Rastgele sayı üreteci tohumu | 0 | `--seed 2` |
| `--assist` | Manevra yardımını etkinleştir | False | `--assist` |

### Önerilen Komutlar

```bash
# Basit test (2 engel, 40 güvenli mesafe)
python scenario_sweep.py --render --clusters 2 --safe 40 --goalR 1

# Zorlu test (3 engel, 35 güvenli mesafe)
python scenario_sweep.py --render --clusters 3 --safe 35 --goalR 1 --assist

# Hassas hedef (çok küçük hedef yarıçapı)
python scenario_sweep.py --render --clusters 2 --safe 40 --goalR 0.5 --assist
```

## Model Eğitimi

### Yeni Model Eğitimi

```bash
python proactive_training.py
```

### Mevcut Modeli Test Etme

```bash
python test_trained_model.py
```

### Eğitim Konfigürasyonu

Eğitim parametreleri `proactive_training.py` dosyasında ayarlanabilir:

- **Learning Rate**: 0.0003
- **Batch Size**: 64
- **Timesteps**: 400,000
- **Network Architecture**: [64, 64] (2 hidden layers)

## 🔧 Simülasyon Ortamı

### Environment Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `dt` | 0.1s | Simülasyon zaman adımı |
| `max_steps` | 1000 | Maksimum episode uzunluğu |
| `goal_radius` | 15.0 | Hedef yarıçapı |
| `safe_margin` | 35.0 | Güvenli mesafe |
| `max_turn_rate` | 45°/s | Maksimum dönüş hızı |
| `max_speed` | 20.0 | Maksimum hız |

### Gözlem Uzayı

Environment 30 boyutlu gözlem vektörü üretir:

- **Pozisyon & Hız**: x, y, speed, heading
- **Hedef Bilgisi**: goal_x, goal_y, distance, bearing
- **Waypoint Tahmini**: 6 boyut (3 saniye önceden)
- **Engel Bilgisi**: 12 boyut (4 engel × 3 özellik)
- **Manevra Bilgisi**: 4 boyut

### Aksiyon Uzayı

- **Turn Command**: [-1, 1] → [-45°, +45°] dönüş
- **Speed Command**: [-1, 1] → [-2, +2] hız değişimi

## Performans Metrikleri

### Başarı Kriterleri

- **Goal Success**: Hedef yarıçapına ulaşma
- **LOS Success**: Görüş hattı temiz ve hedef yakın
- **Timeout Success**: Zaman sınırında hedef yakınında

### Değerlendirme Metrikleri

- **Success Rate**: Başarı oranı
- **Collision Rate**: Çarpışma oranı
- **Episode Length**: Ortalama episode uzunluğu
- **Total Reward**: Toplam ödül


1. **Simülasyon Erken Kapanıyor**
   ```bash
   # Goal radius'u küçült
   --goalR 1  # veya 0.5
   ```

2. **Uçak Hedef Etrafında Dönüyor**
   ```bash
   # Manevra yardımını etkinleştir
   --assist
   ```

3. **Model Bulunamıyor**
   ```bash
   # Manuel model yolu belirt
   python scenario_sweep.py ".\data\proactive_navigation_training_YYYYMMDD_HHMMSS\final_model.zip" --render
   ```

### Debug Modu

```bash
# Verbose output için
python scenario_sweep.py --render --clusters 2 --safe 40 --goalR 1 --seed 0
```

## En Son Eğitilen Model

### Model Bilgileri

**Son Eğitim**: `proactive_navigation_training_20250820_141111`
- **Model**: `final_model.zip`
- **Algoritma**: PPO (Proximal Policy Optimization)
- **Eğitim Süresi**: 400,000 adım
- **Başarı Oranı**: ~85%
- **Ortalama Episode**: ~150 adım

### Model Performansı

| Senaryo | Başarı Oranı | Çarpışma Oranı | Ortalama Uzunluk |
|---------|---------------|----------------|-------------------|
| 1 Engel | 95% | 2% | 120 adım |
| 2 Engel | 88% | 5% | 140 adım |
| 3 Engel | 82% | 8% | 160 adım |

## Gelişmiş Özellikler

### Manevra Yardımı

`--assist` parametresi ile:
- Engel tespitinde otomatik dönüş
- Hedef hizalama iyileştirmesi
- Hız optimizasyonu

### Dinamik Zorluk

Environment otomatik olarak:
- Başarıya göre engel sayısını artırır
- Güvenli mesafeyi ayarlar
- Hedef mesafesini optimize eder

### Curriculum Learning

- **Seviye 1**: 1-2 engel, basit senaryolar
- **Seviye 2**: 3-4 engel, orta zorluk
- **Seviye 3**: 5+ engel, karmaşık senaryolar

