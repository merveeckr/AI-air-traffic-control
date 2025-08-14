# 🚁 AI Navigation & Autonomous Systems

Bu dizin, yapay zeka destekli navigasyon ve otonom sistemler için geliştirilmiş projeleri içerir. Proaktif uçak navigasyonu, canlı simülasyon ve ultra yumuşak risk testi sistemleri bulunmaktadır.

## 📁 Proje Yapısı

```
opensky_demo/
├── live_simulation.py           # Canlı AI simülasyonu
├── proactive_training.py        # Proaktif navigasyon eğitimi
├── proactive_navigation_env.py  # Proaktif navigasyon ortamı
```

##  Proaktif Uçak Navigasyonu

### Proaktif Navigation Environment (`proactive_navigation_env.py`)

#### Özellikler
- **Waypoint Tahmini**: 3 dakika öncesine kadar gelecek rota tahmini
- **Risk Değerlendirmesi**: Proaktif çarpışma kaçınma
- **Dinamik Küme Sistemi**: Rastgele risk kümeleri
- **Gelişmiş Ödül Sistemi**: Proaktif davranış teşviki

#### Observation Space (26 boyut)
- **Mevcut Durum**: [x, y, hız, yön] (4)
- **Tahmin Edilen Waypoint'ler**: [x1, y1, x2, y2, x3, y3] (6)
- **Küme Bilgileri**: [cx, cy, r, risk] × 3 (12)
- **Hedef Bilgileri**: [goal_x, goal_y, mesafe, açı] (4)

#### Action Space (2 boyut)
- **Delta Heading Rate**: [-1.0, 1.0] (dönüş hızı)
- **Delta Speed**: [-1.0, 1.0] (hız değişimi)

### Proaktif Training (`proactive_training.py`)

#### Eğitim Konfigürasyonu
- **Algoritma**: PPO (Proximal Policy Optimization)
- **Toplam Adım**: 500,000
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Paralel Ortam**: 8

#### Model Mimarisi
- **Policy Network**: [256, 256, 128]
- **Value Network**: [256, 256, 128]
- **Activation**: ReLU
- **Initialization**: Orthogonal

#### Callback'ler
- **CheckpointCallback**: Her 50,000 adımda kaydetme
- **EvalCallback**: Her 10,000 adımda değerlendirme

## Canlı Simülasyon (`live_simulation.py`)

### Özellikler
- **Gerçek Zamanlı Görselleştirme**: Her adımda güncellenen harita
- **AI Agent Entegrasyonu**: Eğitilmiş model ile canlı navigasyon
- **Episode Yönetimi**: Çoklu episode simülasyonu
- **Slow Motion**: Detaylı analiz için yavaş çalıştırma

### Simülasyon Seçenekleri
1. **Hızlı Simülasyon**: 3 episode, normal hız
2. **Yavaş Simülasyon**: 3 episode, slow motion
3. **Tek Episode**: Detaylı analiz

### Görselleştirme Özellikleri
- **Risk Kümeleri**: Kırmızı (ana), turuncu (güvenlik tamponu)
- **Waypoint'ler**: Yeşil (güvenli), kırmızı (riskli)
- **AI Rotası**: Mavi çizgi
- **Uçak**: Mavi ok işareti
- **Hedef**: Yeşil yıldız

### Kullanım
```python
python live_simulation.py
```

##  Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements_ai_navigation.txt
```

### Temel Gereksinimler
- `numpy`
- `matplotlib`
- `stable-baselines3`
- `torch`
- `gymnasium`

### Hızlı Başlangıç
```bash
# 1. Proaktif uçak eğitimi
python proactive_training.py

# 2. Canlı simülasyon
python live_simulation.py

# 3. Ortam testi
python proactive_navigation_env.py



## Performans ve Sonuçlar

### Eğitim Sonuçları
- **Başarı Oranı**: %85+ (20 episode test)
- **Proaktif Kaçınma**: %90+ (riskli waypoint'lerden kaçınma)
- **Çarpışma Oranı**: %5- (güvenlik tamponu ile)
- **Eğitim Süresi**: ~30 dakika (500K adım)

### Optimizasyon Özellikleri
- **Gradient Clipping**: max_grad_norm = 0.5
- **Entropy Coefficient**: 0.01 (exploration)
- **GAE Lambda**: 0.95 (advantage estimation)
- **PPO Clip Range**: 0.2 (policy update)

