#  Gelişmiş Hava Risk Analizörü - Advanced Air Risk Analyzer

Bu README, `advanced_air_risk_analyzer.py` dosyası için özel olarak hazırlanmıştır. OpenSky Network API'den gerçek zamanlı uçak verilerini çekip Türkiye hava sahasında gelişmiş risk analizi yapar.

##  Ne Yapıyor?

Bu analizör, OpenSky'dan canlı uçak verilerini alarak:

1. **DBSCAN Kümeleme**: 50km yarıçapında yoğunluk tabanlı kümeleme
2. **Çok Faktörlü Risk Analizi**: Yoğunluk, yükseklik, hız ve çeşitlilik
3. **Uçak Türü Sınıflandırması**: Ticari, askeri, özel, kargo uçakları
4. **Görsel Harita**: Folium ile interaktif risk haritası

## Risk Hesaplama Algoritması

```
Toplam Risk = Yoğunluk(50%) + Yükseklik(20%) + Hız(20%) + Çeşitlilik(10%)
```

### Risk Faktörleri:

- **Yoğunluk Risk (%50)**: Uçak sayısı / Alan (km²)
- **Yükseklik Risk (%20)**: Düşük yükseklik = yüksek risk
- **Hız Risk (%20)**: Yüksek hız = yüksek risk  
- **Çeşitlilik Risk (%10)**: Farklı ülkelerden uçaklar

### Risk Seviyeleri:
- **🔴 YÜKSEK**: > 0.05
- **🟠 ORTA**: 0.02 - 0.05
- **🟢 DÜŞÜK**: < 0.02

##  Harita Görselleştirme

### Küme Gösterimi:
- **Küme Sınırları**: Convex Hull ile çizilen alanlar
- **Risk Merkezleri**: Risk skoruna göre boyut değişen daireler
- **Uçak İkonları**: Türüne göre renkli ikonlar

### Uçak Türleri:
- **✈️ Ticari** (Mavi): Yolcu uçakları
- **🛩️ Askeri** (Yeşil): Askeri uçaklar
- **🛬 Özel** (Turuncu): Özel jetler
- **📦 Kargo** (Gri): Kargo uçakları
- **🚁 Bilinmeyen** (Siyah): Tanımlanamayan

## Kurulum ve Kullanım

### Gereksinimler:
```bash
pip install pandas numpy scikit-learn shapely folium requests
```

### Çalıştırma:
```bash
python advanced_air_risk_analyzer.py
```

### Çıktılar:
- **Terminal**: Detaylı analiz özeti
- **HTML Harita**: `advanced_turkey_air_risk_map_YYYYMMDD_HHMMSS.html`

## 📍 Analiz Bölgesi

**Türkiye Hava Sahası:**
- **Kuzey**: 42°N (Karadeniz)
- **Güney**: 35°N (Akdeniz)
- **Doğu**: 45°E (İran sınırı)
- **Batı**: 25°E (Ege Denizi)

## Örnek Analiz Sonucu

```
 GELİŞMİŞ HAVA RİSK ANALİZİ ÖZETİ
======================================================================
 GENEL BİLGİLER:
   • Toplam Uçak Sayısı: 164
   • Toplam Küme Sayısı: 16
   • Ortalama Küme Başına Uçak: 10.2

  RİSK SEVİYESİ DAĞILIMI:
   • ORTA: 10 küme (62.5%)
   • DÜŞÜK: 6 küme (37.5%)

 EN RİSKLİ KÜMELER:
   • Küme 2: Toplam Risk 0.033 (ORTA) - 63 uçak
     - Yoğunluk: 0.001, Yükseklik: 0.100, Hız: 0.010, Çeşitlilik: 0.100
```

##  Harita Özellikleri

### İnteraktif Elementler:
- **Zoom**: Haritayı yakınlaştırma/uzaklaştırma
- **Popup**: Küme ve uçak detayları
- **Fullscreen**: Tam ekran görüntüleme
- **Renk Kodları**: Risk seviyesine göre renkler

### Görsel Detaylar:
- **Küme Alanları**: Yarı saydam renkli poligonlar
- **Risk Merkezleri**: Risk ile orantılı boyut
- **Uçak Noktaları**: Gerçek zamanlı konumlar
- **Türkiye Sınırları**: Mavi dikdörtgen çerçeve

##  Özelleştirme

### DBSCAN Parametreleri:
```python
epsilon = 50 / kms_per_radian  # 50 km yarıçap
min_samples = max(2, int(len(coords) * 0.02))  # En az 2 uçak
```

### Risk Ağırlıkları:
```python
weights = {
    'density': 0.50,      # Uçak yoğunluğu
    'altitude': 0.20,     # Yükseklik riski
    'velocity': 0.20,     # Hız riski
    'diversity': 0.10     # Çeşitlilik riski
}
```

### Uçak Sınıflandırma:
```python
# Askeri uçaklar
if any(marker in callsign.upper() for marker in ['TUAF', 'TURK', 'MIL', 'FORCE']):
    return 'military'

# Kargo uçakları
if any(marker in callsign.upper() for marker in ['CARGO', 'FREIGHT', 'UPS', 'FEDEX']):
    return 'cargo'
```

##  Kullanım Senaryoları

### Hava Trafik Kontrolü:
- Yoğun bölgeleri tespit etme
- Risk seviyelerini gerçek zamanlı izleme
- Uçak yoğunluğunu analiz etme



##  Bağımlılıklar

- **pandas**: Veri işleme
- **numpy**: Matematiksel hesaplamalar
- **scikit-learn**: DBSCAN kümeleme
- **shapely**: Geometri hesaplamaları
- **folium**: Harita görselleştirme
- **requests**: API veri çekme



