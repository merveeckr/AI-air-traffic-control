# 🚁 Milisaniye Hassasiyetinde Proaktif Navigation Simülasyonu

Bu klasör, AI Agent'ın kümelerin yanına yavaş yavaş yaklaşırken 3 saniye önceden manevra yapmasını milisaniye hassasiyetinde gösteren simülasyonu içerir.

## 🎯 Özellikler

### ⏱️ **Milisaniye Hassasiyeti**
- **Simülasyon Adımı**: 1 milisaniye (0.001s)
- **Simülasyon Hızı**: 0.1x (çok yavaş)
- **Görselleştirme**: 30 FPS

### 🎮 **Proaktif Manevra Sistemi**
- **3 Saniye Önceden Tahmin**: Kümeye yaklaşma süresi 3s veya daha az olduğunda manevra
- **Yavaş Yaklaşma**: Kümelerin yanına çok yavaş yaklaşma
- **Gerçek Zamanlı Analiz**: Her adımda manevra ihtiyacı analizi

### 📊 **Görselleştirme**
- **Küme Durumu**: Her kümenin üzerinde yaklaşma süresi
- **Manevra Uyarısı**: ⚠️ işareti ile 3s önceden uyarı
- **Manevra Geçmişi**: Sarı noktalar ile manevra noktaları
- **Zaman Takibi**: Milisaniye hassasiyetinde zaman gösterimi

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install numpy matplotlib stable-baselines3 torch gymnasium
```

### Çalıştırma
```bash
cd opensky_demo/proactive_live_demo
python millisecond_precision_simulation.py
```

## Simülasyon Senaryosu

###  **Küme Yerleşimi**
- **Küme 1**: (25, 15) - Hedef yolunda, orta risk
- **Küme 2**: (12, 8) - Başlangıç yolunda, düşük risk  
- **Küme 3**: (45, 35) - Hedefin yanında, yüksek risk

###  **Yaklaşma Zamanları**
- **Küme 2**: ~4 saniye sonra yaklaşacak (ilk manevra)
- **Küme 1**: ~8 saniye sonra yaklaşacak
- **Küme 3**: ~12 saniye sonra yaklaşacak

##  Simülasyon Detayları

### **Manevra Tetikleyici**
```python
def predict_maneuver_need(self, cluster):
    """3 saniye önceden manevra ihtiyacını tahmin et"""
    approach_time = self.calculate_cluster_approach_time(cluster)
    return approach_time <= 3.0  # 3 saniye veya daha az
```

### **Yaklaşma Süresi Hesaplama**
```python
def calculate_cluster_approach_time(self, cluster):
    """Kümeye yaklaşma süresini hesapla"""
    distance = np.hypot(cluster['cx'] - self.env.x, cluster['cy'] - self.env.y)
    safe_distance = cluster['r'] + self.env.safe_margin
    approach_distance = distance - safe_distance
    return approach_distance / max(self.env.speed, 0.1)
```

##  Görselleştirme Özellikleri

### **Renk Kodları**
- 🔴 **Kırmızı**: Risk kümeleri
- 🟠 **Turuncu**: Güvenlik tamponu (kesikli çizgi)
- 🔵 **Mavi**: AI Uçak ve rota
- 🟡 **Sarı**: Manevra noktaları
- 🟢 **Yeşil**: Hedef ve güvenli waypoint'ler

### **Bilgi Kutuları**
- **Mavi Kutu**: Simülasyon zamanı (milisaniye)
- **Sarı Kutu**: Son manevra bilgisi
- **Yeşil Kutu**: Hedef mesafesi

## 🎮 Kontroller

### **Simülasyon Kontrolü**
- **Otomatik**: Simülasyon otomatik olarak çalışır
- **Durdurma**: `Ctrl+C` ile durdurulabilir
- **Çıkış**: Simülasyon tamamlandığında otomatik kapanır

### **Görselleştirme Kontrolü**
- **30 FPS**: Akıcı görselleştirme
- **Gerçek Zamanlı**: Her adımda güncellenen harita
- **Zoom**: Matplotlib zoom özellikleri kullanılabilir

##  Çıktı Örnekleri

### **Konsol Çıktısı**
```
🚀 Milisaniye Hassasiyetinde Simülasyon Başlıyor...
⏱️  Simülasyon Hızı: 0.1x
📊 Görselleştirme: 30 FPS

🎯 Yavaş Yaklaşma Senaryosu Oluşturuluyor...
✅ Yavaş yaklaşma senaryosu oluşturuldu:
   Küme 1: (25.0, 15.0) - Risk: 0.6 - Yaklaşma: 8.0s
   Küme 2: (12.0, 8.0) - Risk: 0.4 - Yaklaşma: 4.0s
   Küme 3: (45.0, 35.0) - Risk: 0.8 - Yaklaşma: 12.0s
```

### **Görsel Çıktı**
- **Harita**: Gerçek zamanlı güncellenen navigasyon haritası
- **Zaman**: Milisaniye hassasiyetinde zaman gösterimi
- **Manevra**: 3 saniye önceden manevra uyarıları

##  Özelleştirme

### **Simülasyon Hızı**
```python
self.simulation_speed = 0.1  # 0.1x hız (çok yavaş)
# Daha hızlı için: 0.5, 1.0, 2.0
```

### **Manevra Tetikleyici Süresi**
```python
return approach_time <= 3.0  # 3 saniye önceden
# Daha erken uyarı için: 5.0, 7.0
```

### **Görselleştirme FPS**
```python
self.visualization_fps = 30  # 30 FPS
# Daha akıcı için: 60, daha yavaş için: 15
```


## Teknik Detaylar

### **Milisaniye Hassasiyeti**
- **dt**: 0.001 saniye (1 milisaniye)
- **Hassasiyet**: 3 ondalık basamak
- **Zaman Takibi**: `current_time` ile sürekli güncelleme

### **Manevra Analizi**
- **Her Adımda**: Tüm kümeler için yaklaşma süresi hesaplanır
- **3s Tetikleyici**: Yaklaşma süresi ≤ 3s olduğunda manevra
- **Geçmiş Takibi**: Manevra noktaları sarı noktalarla işaretlenir


