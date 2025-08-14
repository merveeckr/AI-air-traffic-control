# OpenSky API Demo

Bu klasör, **OpenSky Network API**'den gerçek zamanlı uçak verilerini çekip işleyebildiğimizi gösteren demo dosyalarını içerir.

## **Ne Yapıyor?**

- **Gerçek zamanlı uçak verilerini** OpenSky'den çeker
- **Türkiye hava sahasındaki** aktif uçakları listeler
- **Uçak bilgilerini** (konum, hız, yön, yükseklik) gösterir
- **Veri işleme** ve **filtreleme** örnekleri sunar
- **JSON formatında** veri kaydeder

### **1. Gereksinimler**
```bash
pip install -r requirements.txt
```

### **2. Çalıştır**
```bash
python opensky_demo.py
```

### **3. Sonuç**
- Terminal'de aktif uçak listesi
- `sample_data/` klasöründe JSON dosyası
- Detaylı uçak bilgileri

## **OpenSky'den Gelen Veriler**

### **Temel Bilgiler:**
- **ICAO24**: Uçak kimlik kodu
- **Latitude/Longitude**: Konum koordinatları
- **Altitude**: Yükseklik (feet)
- **Velocity**: Hız (m/s)
- **Heading**: Yön (derece)
- **Vertical Rate**: Dikey hız (m/s)
- **Squawk**: Transponder kodu
- **Country**: Ülke

### **Örnek Veri:**
```json
{
  "icao24": "a4b123",
  "callsign": "THY123",
  "origin_country": "Turkey",
  "time_position": 1640995200,
  "last_contact": 1640995200,
  "longitude": 28.9784,
  "latitude": 41.0082,
  "baro_altitude": 30000,
  "on_ground": false,
  "velocity": 250.0,
  "true_track": 45.0,
  "vertical_rate": 0.0,
  "squawk": "1234"
}
```

## **API Özellikleri**

### **Veri Limitleri:**
- **Ücretsiz hesap**: 1000 istek/gün
- **Kayıtlı hesap**: 10000 istek/gün
- **Premium hesap**: Sınırsız

### **Güncelleme Sıklığı:**
- **ADS-B**: Her 5-10 saniyede
- **MLAT**: Her 10-30 saniyede
- **TIS-B**: Her 5-15 saniyede

## **Türkiye Hava Sahası**

### **Koordinat Sınırları:**
- **Kuzey**: 42.0° N
- **Güney**: 35.0° N  
- **Doğu**: 45.0° E
- **Batı**: 25.0° E

### **Önemli Havaalanları:**
- **İstanbul (IST)**: 41.0082°N, 28.9784°E
- **Ankara (ESB)**: 39.9334°N, 32.8597°E
- **İzmir (ADB)**: 38.4192°N, 27.1428°E
- **Antalya (AYT)**: 36.8969°N, 30.7133°E

## **Hata Durumları**

### **Yaygın Hatalar:**
- **Rate Limit**: Çok fazla istek
- **Authentication**: API key hatası
- **Network**: İnternet bağlantı sorunu
- **No Data**: Belirtilen bölgede uçak yok

### **Çözümler:**
- İstekleri yavaşlat
- API key'i kontrol et
- İnternet bağlantısını kontrol et
- Farklı bölge dene

##  **Kaynaklar**

- [OpenSky Network API](https://opensky-network.org/apidocs/)
- [API Dokümantasyonu](https://opensky-network.org/apidocs/)
- [Python Client](https://github.com/openskynetwork/opensky-python)
- [Veri Formatı](https://opensky-network.org/data/datasets)


