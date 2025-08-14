#!/usr/bin/env python3
"""
OpenSky API Demo - Gerçek Zamanlı Uçak Verilerini Çek ve İşle.

Bu dosya OpenSky Network API'den uçak verilerini çekip işleyebildiğimizi gösterir.
"""

# Standard library imports
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Third-party imports
import requests

# Local imports
sys.path.append('..')
from src.opensky_client import OpenSkyClient


class OpenSkyDemo:
    """OpenSky API demo sınıfı - veri çekme ve işleme örnekleri."""
    
    def __init__(self):
        """OpenSky demo sınıfını başlat."""
        self.base_url = "https://opensky-network.org/api"
        self.username = None  # OpenSky kullanıcı adı (opsiyonel)
        self.password = None  # OpenSky şifresi (opsiyonel)
        
        # Türkiye hava sahası koordinatları
        self.turkey_bbox = {
            'min_lat': 35.0,   # Güney sınırı
            'max_lat': 42.0,   # Kuzey sınırı
            'min_lon': 25.0,   # Batı sınırı
            'max_lon': 45.0    # Doğu sınırı
        }
        
        # Örnek veri klasörü
        self.sample_data_dir = "sample_data"
        os.makedirs(self.sample_data_dir, exist_ok=True)
    
    def set_credentials(self, username: str, password: str):
        """OpenSky hesap bilgilerini ayarla (opsiyonel)."""
        self.username = username
        self.password = password
        print(f"OpenSky hesabı ayarlandı: {username}")
    
    def get_turkey_aircraft(self) -> Optional[List[Dict]]:
        """Türkiye hava sahasındaki aktif uçakları getir."""
        try:
            # API endpoint
            url = f"{self.base_url}/states/all"
            
            # Query parametreleri
            params = {
                'lamin': self.turkey_bbox['min_lat'],
                'lamax': self.turkey_bbox['max_lat'],
                'lomin': self.turkey_bbox['min_lon'],
                'lomax': self.turkey_bbox['max_lon']
            }
            
            # Authentication (eğer varsa)
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            print("Türkiye hava sahasından uçak verileri çekiliyor...")
            
            # API isteği
            response = requests.get(url, params=params, auth=auth, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'states' in data and data['states']:
                print(f" {len(data['states'])} uçak bulundu!")
                return data['states']
            else:
                print(" Belirtilen bölgede aktif uçak bulunamadı.")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f" API isteği hatası: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f" JSON parse hatası: {e}")
            return None
        except Exception as e:
            print(f" Beklenmeyen hata: {e}")
            return None
    
    def process_aircraft_data(self, raw_states: List[List]) -> List[Dict]:
        """Ham uçak verilerini işle ve anlamlı hale getir."""
        processed_aircraft = []
        
        # OpenSky veri formatı (index'ler)
        # 0: icao24, 1: callsign, 2: origin_country, 3: time_position, 
        # 4: last_contact, 5: longitude, 6: latitude, 7: baro_altitude,
        # 8: on_ground, 9: velocity, 10: true_track, 11: vertical_rate,
        # 12: squawk, 13: spi, 14: position_source
        
        for state in raw_states:
            if len(state) >= 15:  # Minimum veri kontrolü
                aircraft = {
                    'icao24': state[0] or 'Unknown',
                    'callsign': state[1] or 'Unknown',
                    'origin_country': state[2] or 'Unknown',
                    'time_position': state[3],
                    'last_contact': state[4],
                    'longitude': state[5],
                    'latitude': state[6],
                    'baro_altitude': state[7],
                    'on_ground': state[8],
                    'velocity': state[9],
                    'true_track': state[10],
                    'vertical_rate': state[11],
                    'squawk': state[12],
                    'spi': state[13],
                    'position_source': state[14]
                }
                
                # Zaman bilgilerini formatla
                if aircraft['time_position']:
                    aircraft['time_position_formatted'] = datetime.fromtimestamp(
                        aircraft['time_position']
                    ).strftime('%Y-%m-%d %H:%M:%S')
                
                if aircraft['last_contact']:
                    aircraft['last_contact_formatted'] = datetime.fromtimestamp(
                        aircraft['last_contact']
                    ).strftime('%Y-%m-%d %H:%M:%S')
                
                # Yükseklik bilgisini feet'ten metre'ye çevir
                if aircraft['baro_altitude']:
                    aircraft['altitude_meters'] = aircraft['baro_altitude'] * 0.3048
                
                # Hız bilgisini m/s'den km/h'ye çevir
                if aircraft['velocity']:
                    aircraft['velocity_kmh'] = aircraft['velocity'] * 3.6
                
                processed_aircraft.append(aircraft)
        
        return processed_aircraft
    
    def filter_aircraft(self, aircraft_list: List[Dict], **filters) -> List[Dict]:
        """Uçak listesini belirli kriterlere göre filtrele."""
        filtered = aircraft_list
        
        # Yükseklik filtresi
        if 'min_altitude' in filters:
            filtered = [a for a in filtered if a.get('baro_altitude') is not None and a.get('baro_altitude', 0) >= filters['min_altitude']]
        
        if 'max_altitude' in filters:
            filtered = [a for a in filtered if a.get('baro_altitude') is not None and a.get('baro_altitude', float('inf')) <= filters['max_altitude']]
        
        # Hız filtresi
        if 'min_velocity' in filters:
            filtered = [a for a in filtered if a.get('velocity') is not None and a.get('velocity', 0) >= filters['min_velocity']]
        
        if 'max_velocity' in filters:
            filtered = [a for a in filtered if a.get('velocity') is not None and a.get('velocity', float('inf')) <= filters['max_velocity']]
        
        # Ülke filtresi
        if 'country' in filters:
            filtered = [a for a in filtered if a.get('origin_country', '').lower() == filters['country'].lower()]
        
        # Yerde olan uçakları filtrele
        if 'in_air_only' in filters and filters['in_air_only']:
            filtered = [a for a in filtered if not a.get('on_ground', True)]
        
        return filtered
    
    def analyze_aircraft_data(self, aircraft_list: List[Dict]) -> Dict:
        """Uçak verilerini analiz et ve istatistikler çıkar."""
        if not aircraft_list:
            return {}
        
        analysis = {
            'total_aircraft': len(aircraft_list),
            'countries': {},
            'altitude_stats': {},
            'velocity_stats': {},
            'position_sources': {},
            'ground_vs_air': {'ground': 0, 'air': 0}
        }
        
        altitudes = []
        velocities = []
        
        for aircraft in aircraft_list:
            # Ülke sayısı
            country = aircraft.get('origin_country', 'Unknown')
            analysis['countries'][country] = analysis['countries'].get(country, 0) + 1
            
            # Yükseklik istatistikleri
            if aircraft.get('baro_altitude'):
                altitudes.append(aircraft['baro_altitude'])
            
            # Hız istatistikleri
            if aircraft.get('velocity'):
                velocities.append(aircraft['velocity'])
            
            # Pozisyon kaynağı
            source = aircraft.get('position_source', 'Unknown')
            analysis['position_sources'][source] = analysis['position_sources'].get(source, 0) + 1
            
            # Yerde/Havada sayısı
            if aircraft.get('on_ground'):
                analysis['ground_vs_air']['ground'] += 1
            else:
                analysis['ground_vs_air']['air'] += 1
        
        # Yükseklik istatistikleri
        if altitudes:
            analysis['altitude_stats'] = {
                'min': min(altitudes),
                'max': max(altitudes),
                'avg': sum(altitudes) / len(altitudes)
            }
        
        # Hız istatistikleri
        if velocities:
            analysis['velocity_stats'] = {
                'min': min(velocities),
                'max': max(velocities),
                'avg': sum(velocities) / len(velocities)
            }
        
        return analysis
    
    def save_to_json(self, data: Dict, filename: str = None) -> str:
        """Veriyi JSON dosyasına kaydet."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"turkey_aircraft_{timestamp}.json"
        
        filepath = os.path.join(self.sample_data_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f" Veri kaydedildi: {filepath}")
            return filepath
        except Exception as e:
            print(f" Dosya kaydetme hatası: {e}")
            return None
    
    def display_aircraft_info(self, aircraft_list: List[Dict], limit: int = 10):
        """Uçak bilgilerini terminal'de göster."""
        if not aircraft_list:
            print(" Gösterilecek uçak bulunamadı.")
            return
        
        print(f"\n İlk {min(limit, len(aircraft_list))} Uçak Bilgisi:")
        print("=" * 80)
        
        for i, aircraft in enumerate(aircraft_list[:limit]):
            print(f"\n{i+1}. Uçak:")
            print(f"   ICAO24: {aircraft.get('icao24', 'N/A')}")
            print(f"   Callsign: {aircraft.get('callsign', 'N/A')}")
            print(f"   Ülke: {aircraft.get('origin_country', 'N/A')}")
            print(f"   Konum: ({aircraft.get('latitude', 'N/A'):.4f}, {aircraft.get('longitude', 'N/A'):.4f})")
            print(f"   Yükseklik: {aircraft.get('baro_altitude', 'N/A')} ft ({aircraft.get('altitude_meters', 'N/A'):.0f} m)")
            print(f"   Hız: {aircraft.get('velocity', 'N/A')} m/s ({aircraft.get('velocity_kmh', 'N/A'):.0f} km/h)")
            print(f"   Yön: {aircraft.get('true_track', 'N/A')}°")
            print(f"   Squawk: {aircraft.get('squawk', 'N/A')}")
            print(f"   Durum: {'Yerde' if aircraft.get('on_ground') else 'Havada'}")
            
            if aircraft.get('time_position_formatted'):
                print(f"   Son Pozisyon: {aircraft['time_position_formatted']}")
    
    def display_analysis(self, analysis: Dict):
        """Analiz sonuçlarını göster."""
        if not analysis:
            print("Analiz verisi bulunamadı.")
            return
        
        print(f"\nVeri Analizi:")
        print("=" * 50)
        print(f"Toplam Uçak: {analysis.get('total_aircraft', 0)}")
        
        # Ülke dağılımı
        if analysis.get('countries'):
            print(f"\n Ülke Dağılımı:")
            for country, count in sorted(analysis['countries'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {country}: {count} uçak")
        
        # Yükseklik istatistikleri
        if analysis.get('altitude_stats'):
            alt_stats = analysis['altitude_stats']
            print(f"\n Yükseklik İstatistikleri:")
            print(f"   Min: {alt_stats['min']:.0f} ft")
            print(f"   Max: {alt_stats['max']:.0f} ft")
            print(f"   Ortalama: {alt_stats['avg']:.0f} ft")
        
        # Hız istatistikleri
        if analysis.get('velocity_stats'):
            vel_stats = analysis['velocity_stats']
            print(f"\n Hız İstatistikleri:")
            print(f"   Min: {vel_stats['min']:.1f} m/s")
            print(f"   Max: {vel_stats['max']:.1f} m/s")
            print(f"   Ortalama: {vel_stats['avg']:.1f} m/s")
        
        # Yerde/Havada dağılımı
        ground_air = analysis.get('ground_vs_air', {})
        if ground_air:
            print(f"\n Durum Dağılımı:")
            print(f"   Yerde: {ground_air['ground']} uçak")
            print(f"   Havada: {ground_air['air']} uçak")
    
    def run_demo(self):
        """Ana demo fonksiyonu."""
        print(" OpenSky API Demo Başlıyor!")
        print("=" * 50)
        
        # 1. Türkiye hava sahasından uçak verilerini çek
        raw_states = self.get_turkey_aircraft()
        
        if raw_states is None:
            print(" Demo çalıştırılamadı. API hatası.")
            return
        
        if not raw_states:
            print(" Aktif uçak bulunamadı. Demo tamamlandı.")
            return
        
        # 2. Verileri işle
        print("\n Veriler işleniyor...")
        processed_aircraft = self.process_aircraft_data(raw_states)
        
        # 3. Veri analizi
        print(" Veri analizi yapılıyor...")
        analysis = self.analyze_aircraft_data(processed_aircraft)
        
        # 4. Sonuçları göster
        self.display_aircraft_info(processed_aircraft, limit=5)
        self.display_analysis(analysis)
        
        # 5. Filtreleme örnekleri
        print(f"\n Filtreleme Örnekleri:")
        print("=" * 40)
        
        # Sadece havadaki uçaklar
        in_air = self.filter_aircraft(processed_aircraft, in_air_only=True)
        print(f" Havadaki uçaklar: {len(in_air)}")
        
        # Yüksek irtifadaki uçaklar (30000 ft üzeri)
        high_alt = self.filter_aircraft(processed_aircraft, min_altitude=30000)
        print(f" Yüksek irtifa (30k+ ft): {len(high_alt)}")
        
        # Hızlı uçaklar (200+ m/s)
        fast_aircraft = self.filter_aircraft(processed_aircraft, min_velocity=200)
        print(f" Hızlı uçaklar (200+ m/s): {len(fast_aircraft)}")
        
        # 6. Veriyi kaydet
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'total_aircraft': len(processed_aircraft),
            'aircraft': processed_aircraft,
            'analysis': analysis,
            'filters': {
                'in_air': len(in_air),
                'high_altitude': len(high_alt),
                'fast': len(fast_aircraft)
            }
        }
        
        saved_file = self.save_to_json(data_to_save)
        
        if saved_file:
            print(f"\n Demo tamamlandı! Veriler kaydedildi: {saved_file}")
        else:
            print("\n Demo tamamlandı ancak veri kaydedilemedi.")


def main():
    """Ana fonksiyon."""
    print(" OpenSky API Demo")
    print("=" * 30)
    
    # Demo nesnesini oluştur
    demo = OpenSkyDemo()
    
    # Opsiyonel: OpenSky hesap bilgilerini ayarla
    # demo.set_credentials("your_username", "your_password")
    
    # Demo'yu çalıştır
    demo.run_demo()
    
    print("\n Demo tamamlandı!")
    print(" Veriler 'sample_data/' klasöründe kaydedildi.")


if __name__ == "__main__":
    main()
