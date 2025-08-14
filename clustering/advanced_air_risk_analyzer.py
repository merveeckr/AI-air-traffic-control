"""
Gelişmiş Hava Risk Analizörü
Gerçek uçak ikonları, hareket simülasyonu ve interaktif özellikler
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point
import folium
from folium import plugins
import requests
import time
from datetime import datetime
import json

class AdvancedAirRiskAnalyzer:
    def __init__(self):
        """Gelişmiş risk analizörünü başlat"""
        self.turkey_bounds = {
            'lamin': 35, 'lomin': 25,  # Güneybatı
            'lamax': 42, 'lomax': 45   # Kuzeydoğu
        }
        self.opensky_url = "https://opensky-network.org/api/states/all"
        
        # Uçak türleri ve ikonları
        self.aircraft_icons = {
            'commercial': '✈️',      # Ticari uçak
            'military': '🛩️',       # Askeri uçak
            'private': '🛬',         # Özel uçak
            'cargo': '📦',           # Kargo uçak
            'unknown': '🚁'          # Bilinmeyen
        }
        
    def fetch_live_data(self):
        """OpenSky'dan canlı veri çek"""
        print("OpenSky'dan canlı veri çekiliyor...")
        
        try:
            params = {
                'lamin': self.turkey_bounds['lamin'],
                'lomin': self.turkey_bounds['lomin'],
                'lamax': self.turkey_bounds['lamax'],
                'lomax': self.turkey_bounds['lomax']
            }
            
            response = requests.get(self.opensky_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            states = data.get("states", [])
            
            if not states:
                print("Veri bulunamadı")
                return None
            
            print(f" {len(states)} uçak verisi alındı")
            
            # DataFrame'e çevir
            df = pd.DataFrame(states, columns=[
                'icao24', 'callsign', 'origin_country', 'time_position', 'last_contact',
                'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
                'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk',
                'spi', 'position_source'
            ])
            
            # Eksik koordinatları temizle
            df = df.dropna(subset=['latitude', 'longitude'])
            
            # Sadece havadaki uçakları al
            df = df[df['on_ground'] == False]
            
            # Uçak türünü belirle
            df['aircraft_type'] = df.apply(self._classify_aircraft, axis=1)
            
            print(f" {len(df)} havadaki uçak filtrelendi")
            return df
            
        except Exception as e:
            print(f" Veri çekme hatası: {e}")
            return None
    
    def _classify_aircraft(self, row):
        """Uçak türünü belirle"""
        callsign = str(row['callsign']).strip()
        country = str(row['origin_country']).strip()
        
        # Askeri uçaklar
        if any(marker in callsign.upper() for marker in ['TUAF', 'TURK', 'MIL', 'FORCE']):
            return 'military'
        
        # Kargo uçakları
        if any(marker in callsign.upper() for marker in ['CARGO', 'FREIGHT', 'UPS', 'FEDEX']):
            return 'cargo'
        
        # Özel uçaklar
        if len(callsign) <= 3 or callsign.startswith('N') or callsign.startswith('G'):
            return 'private'
        
        # Ticari uçaklar (varsayılan)
        return 'commercial'
    
    def perform_clustering(self, df):
        """DBSCAN ile kümeleme yap"""
        print(" DBSCAN kümeleme başlatılıyor...")
        
        if len(df) < 2:
            print(" Kümeleme için yeterli veri yok")
            return df
        
        coords = df[['latitude', 'longitude']].to_numpy()
        
        # DBSCAN parametreleri
        kms_per_radian = 6371.0088
        epsilon = 50 / kms_per_radian  # 50 km yarıçap
        min_samples = max(2, int(len(coords) * 0.02))
        
        print(f" DBSCAN parametreleri: eps={50}km, min_samples={min_samples}")
        
        db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
        df['cluster'] = db.labels_
        
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        noise_count = (db.labels_ == -1).sum()
        
        print(f" {n_clusters} küme oluşturuldu")
        if noise_count > 0:
            print(f" {noise_count} gürültü noktası tespit edildi")
        
        return df
    
    def calculate_advanced_risk_scores(self, df):
        """Gelişmiş risk skorları hesapla"""
        print(" Gelişmiş risk skorları hesaplanıyor...")
        
        risk_data = []
        
        for cluster_id in set(df['cluster']):
            if cluster_id == -1:
                continue
                
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_points = cluster_df[['latitude', 'longitude']].to_numpy()
            num_planes = len(cluster_points)
            
            # Küme alanını hesapla
            try:
                hull = MultiPoint([Point(p[1], p[0]) for p in cluster_points]).convex_hull
                area_km2 = max(1, hull.area * (111**2))
            except:
                area_km2 = 1
            
            # Temel risk skoru
            density_risk = num_planes / area_km2
            
            # Gelişmiş risk faktörleri
            altitude_risk = self._calculate_altitude_risk(cluster_df)
            velocity_risk = self._calculate_velocity_risk(cluster_df)
            diversity_risk = self._calculate_diversity_risk(cluster_df)
            
            # Toplam risk skoru (ağırlıklı)
            total_risk = (
                density_risk * 0.5 +      # Yoğunluk %50
                altitude_risk * 0.2 +     # Yükseklik %20
                velocity_risk * 0.2 +     # Hız %20
                diversity_risk * 0.1      # Çeşitlilik %10
            )
            
            # Risk seviyesi
            if total_risk > 0.05:
                risk_level = "YÜKSEK"
                color = 'red'
            elif total_risk > 0.02:
                risk_level = "ORTA"
                color = 'orange'
            else:
                risk_level = "DÜŞÜK"
                color = 'green'
            
            # Küme merkezi
            centroid = [cluster_points[:,0].mean(), cluster_points[:,1].mean()]
            
            risk_data.append({
                'cluster_id': cluster_id,
                'num_planes': num_planes,
                'area_km2': area_km2,
                'density_risk': density_risk,
                'altitude_risk': altitude_risk,
                'velocity_risk': velocity_risk,
                'diversity_risk': diversity_risk,
                'total_risk': total_risk,
                'risk_level': risk_level,
                'color': color,
                'centroid': centroid,
                'points': cluster_points,
                'aircraft_data': cluster_df
            })
        
        print(f" {len(risk_data)} küme için gelişmiş risk skorları hesaplandı")
        return risk_data
    
    def _calculate_altitude_risk(self, cluster_df):
        """Yükseklik riski hesapla"""
        altitudes = cluster_df['baro_altitude'].dropna()
        if len(altitudes) == 0:
            return 0
        
        # Düşük yükseklik = yüksek risk
        avg_altitude = altitudes.mean()
        if avg_altitude < 5000:  # 5000 ft altı
            return 0.1
        elif avg_altitude < 10000:  # 5000-10000 ft
            return 0.05
        else:
            return 0.01
    
    def _calculate_velocity_risk(self, cluster_df):
        """Hız riski hesapla"""
        velocities = cluster_df['velocity'].dropna()
        if len(velocities) == 0:
            return 0
        
        # Yüksek hız = yüksek risk
        avg_velocity = velocities.mean()
        if avg_velocity > 250:  # 250+ m/s
            return 0.1
        elif avg_velocity > 200:  # 200-250 m/s
            return 0.05
        else:
            return 0.01
    
    def _calculate_diversity_risk(self, cluster_df):
        """Çeşitlilik riski hesapla"""
        countries = cluster_df['origin_country'].dropna()
        if len(countries) == 0:
            return 0
        
        # Farklı ülkelerden uçaklar = yüksek risk
        unique_countries = countries.nunique()
        if unique_countries > 5:
            return 0.1
        elif unique_countries > 3:
            return 0.05
        else:
            return 0.01
    
    def create_advanced_risk_map(self, df, risk_data):
        """Gelişmiş risk haritası oluştur"""
        print(" Gelişmiş risk haritası oluşturuluyor...")
        
        center_lat = (self.turkey_bounds['lamin'] + self.turkey_bounds['lamax']) / 2
        center_lon = (self.turkey_bounds['lomin'] + self.turkey_bounds['lomax']) / 2
        
        # Harita oluştur
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Türkiye sınırları
        turkey_bounds = [
            [self.turkey_bounds['lamin'], self.turkey_bounds['lomin']],
            [self.turkey_bounds['lamax'], self.turkey_bounds['lomax']]
        ]
        
        folium.Rectangle(
            bounds=turkey_bounds,
            color='blue',
            weight=2,
            fill=False,
            opacity=0.8,
            popup="Türkiye Hava Sahası"
        ).add_to(m)
        
        # Her küme için
        for cluster_info in risk_data:
            # Küme sınırları (Convex Hull)
            try:
                hull = MultiPoint([Point(p[1], p[0]) for p in cluster_info['points']]).convex_hull
                hull_coords = list(hull.exterior.coords)
                hull_coords = [[coord[1], coord[0]] for coord in hull_coords]
                
                folium.Polygon(
                    locations=hull_coords,
                    color=cluster_info['color'],
                    weight=2,
                    fill=True,
                    fillColor=cluster_info['color'],
                    fillOpacity=0.1,
                    popup=f"Küme {cluster_info['cluster_id']} - {cluster_info['risk_level']}"
                ).add_to(m)
            except:
                pass
            
            # Küme merkezi
            folium.CircleMarker(
                location=cluster_info['centroid'],
                radius=15 + cluster_info['total_risk'] * 200,
                color=cluster_info['color'],
                fill=True,
                fill_opacity=0.8,
                popup=f"""
                <b>Küme {cluster_info['cluster_id']}</b><br>
                Uçak Sayısı: {cluster_info['num_planes']}<br>
                Alan: {cluster_info['area_km2']:.1f} km²<br>
                Yoğunluk Risk: {cluster_info['density_risk']:.3f}<br>
                Yükseklik Risk: {cluster_info['altitude_risk']:.3f}<br>
                Hız Risk: {cluster_info['velocity_risk']:.3f}<br>
                Çeşitlilik Risk: {cluster_info['diversity_risk']:.3f}<br>
                <b>Toplam Risk: {cluster_info['total_risk']:.3f}</b><br>
                Risk Seviyesi: {cluster_info['risk_level']}
                """
            ).add_to(m)
            
            # Küme içindeki uçaklar (gerçek ikonlarla)
            for _, aircraft in cluster_info['aircraft_data'].iterrows():
                icon_type = aircraft['aircraft_type']
                icon = self.aircraft_icons.get(icon_type, self.aircraft_icons['unknown'])
                
                # Uçak ikonu
                folium.Marker(
                    location=[aircraft['latitude'], aircraft['longitude']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 20px;">{icon}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    ),
                    popup=f"""
                    <b>{aircraft['callsign']}</b><br>
                    ICAO: {aircraft['icao24']}<br>
                    Ülke: {aircraft['origin_country']}<br>
                    Tür: {icon_type}<br>
                    Yükseklik: {aircraft['baro_altitude']:.0f} ft<br>
                    Hız: {aircraft['velocity']:.0f} m/s<br>
                    Küme: {cluster_info['cluster_id']}
                    """
                ).add_to(m)
        
        # Gelişmiş açıklama
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Seviyeleri:</b></p>
        <p><span style="color:red;">🔴 YÜKSEK</span> > 0.05</p>
        <p><span style="color:orange;">🟠 ORTA</span> 0.02-0.05</p>
        <p><span style="color:green;">🟢 DÜŞÜK</span> < 0.02</p>
        <p><b>Uçak Türleri:</b></p>
        <p>✈️ Ticari | 🛩️ Askeri | 🛬 Özel | 📦 Kargo | 🚁 Bilinmeyen</p>
        <p><small>Risk = Yoğunluk(50%) + Yükseklik(20%) + Hız(20%) + Çeşitlilik(10%)</small></p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Fullscreen butonu
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def print_advanced_analysis_summary(self, risk_data):
        """Gelişmiş analiz özetini yazdır"""
        if not risk_data:
            return
        
        print("\n" + "="*70)
        print("🚁 GELİŞMİŞ HAVA RİSK ANALİZİ ÖZETİ")
        print("="*70)
        
        total_planes = sum(cluster['num_planes'] for cluster in risk_data)
        total_clusters = len(risk_data)
        
        print(f"\n GENEL BİLGİLER:")
        print(f"   • Toplam Uçak Sayısı: {total_planes}")
        print(f"   • Toplam Küme Sayısı: {total_clusters}")
        print(f"   • Ortalama Küme Başına Uçak: {total_planes/total_clusters:.1f}")
        
        # Risk seviyelerine göre dağılım
        risk_levels = [cluster['risk_level'] for cluster in risk_data]
        risk_distribution = pd.Series(risk_levels).value_counts()
        
        print(f"\n  RİSK SEVİYESİ DAĞILIMI:")
        for level, count in risk_distribution.items():
            percentage = (count / total_clusters) * 100
            print(f"   • {level}: {count} küme ({percentage:.1f}%)")
        
        # En riskli kümeler
        print(f"\n EN RİSKLİ KÜMELER:")
        top_risky = sorted(risk_data, key=lambda x: x['total_risk'], reverse=True)[:3]
        for cluster in top_risky:
            print(f"   • Küme {cluster['cluster_id']}: "
                  f"Toplam Risk {cluster['total_risk']:.3f} "
                  f"({cluster['risk_level']}) - "
                  f"{cluster['num_planes']} uçak")
            print(f"     - Yoğunluk: {cluster['density_risk']:.3f}, "
                  f"Yükseklik: {cluster['altitude_risk']:.3f}, "
                  f"Hız: {cluster['velocity_risk']:.3f}, "
                  f"Çeşitlilik: {cluster['diversity_risk']:.3f}")
        
        print("\n" + "="*70)
    
    def run_analysis(self):
        """Tam analizi çalıştır"""
        print(" Gelişmiş Hava Risk Analizi Başlatılıyor...")
        print(f" Analiz Bölgesi: Türkiye ({self.turkey_bounds['lamin']}°N-{self.turkey_bounds['lamax']}°N, "
              f"{self.turkey_bounds['lomin']}°E-{self.turkey_bounds['lomax']}°E)")
        
        # 1. Canlı veri çek
        df = self.fetch_live_data()
        if df is None or len(df) == 0:
            print(" Analiz için veri bulunamadı")
            return False
        
        # 2. Kümeleme yap
        df = self.perform_clustering(df)
        
        # 3. Gelişmiş risk skorlarını hesapla
        risk_data = self.calculate_advanced_risk_scores(df)
        
        # 4. Sonuçları yazdır
        self.print_advanced_analysis_summary(risk_data)
        
        # 5. Gelişmiş harita oluştur
        risk_map = self.create_advanced_risk_map(df, risk_data)
        
        # 6. Haritayı kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"advanced_turkey_air_risk_map_{timestamp}.html"
        risk_map.save(output_file)
        
        print(f"\n Gelişmiş analiz tamamlandı!")
        print(f" Harita kaydedildi: {output_file}")
        
        return True

def main():
    """Ana fonksiyon"""
    analyzer = AdvancedAirRiskAnalyzer()
    success = analyzer.run_analysis()
    
    if not success:
        print("\n Analiz başarısız oldu")

if __name__ == "__main__":
    main()
