import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_centrifuge_data(excel_path: str) -> pd.DataFrame:
    """
    Lädt und verarbeitet die GEA Zentrifugen-Daten aus Excel
    
    Args:
        excel_path: Pfad zur Excel-Datei
    
    Returns:
        DataFrame mit ML-ready Zentrifugen-Daten
    """
    
    print(f"🔄 Lade Zentrifugen-Daten aus: {excel_path}")
    
    try:
        # Excel laden
        df = pd.read_excel(excel_path, sheet_name='Ausgewählte LISTE - Final')
        print(f"✅ {len(df)} Zentrifugen-Datensätze geladen")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der Excel-Datei: {e}")
        # Fallback auf Mock-Daten
        return create_mock_centrifuge_data()
    
    # Spalten-Mapping für ML-Kompatibilität
    column_mapping = {
        'Application': 'category',
        'Sub Application': 'subcategory', 
        'SEP_SQLLangtyp': 'model',
        'Listprice': 'purchase_price',
        'SEP_SQLMotorPowerKW': 'motor_power_kw',
        'power consumption TOTAL [kW]': 'total_power_consumption',
        'SEP_SQLOpWaterls': 'water_consumption_ls',
        'SEP_SQLOpWaterliteject': 'water_per_ejection',
        'SEP_DriveType': 'drive_type',
        'SEP_Level': 'quality_level',
        'ejection system': 'ejection_system',
        'SEP_CapacityMinInp': 'capacity_min',
        'SEP_CapacityMaxInp': 'capacity_max',
        'SEP_SQLTotalWeightKg': 'total_weight_kg',
        'SEP_SQLLength': 'length_mm',
        'SEP_SQLWidth': 'width_mm',
        'SEP_SQLHeigth': 'height_mm'
    }
    
    # Fehlende Spalten hinzufügen und mappen
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Zusätzliche Standard-Spalten für ML hinzufügen
    df['manufacturer'] = 'GEA'  # Alle sind GEA-Maschinen
    df['age_years'] = np.random.uniform(0.5, 5, len(df))  # Mock-Alter
    df['warranty_years'] = 2  # Standard GEA-Garantie
    df['expected_lifetime'] = 15  # Typische Lebensdauer Zentrifugen
    
    # Standorte zuweisen (realistisch für GEA)
    locations = ['Düsseldorf (HQ)', 'Oelde', 'Berlin', 'Hamburg', 'Kopenhagen', 'Mailand']
    df['location'] = np.random.choice(locations, len(df))
    
    # Usage Patterns basierend auf Anwendung
    usage_mapping = {
        'Citrus': 'Extended (12h/Tag)',
        'Wine': 'Standard (8h/Tag)', 
        'Dairy': '24/7 Betrieb',
        'Industrial': '24/7 Betrieb'
    }
    df['usage_pattern'] = df['category'].map(usage_mapping).fillna('Standard (8h/Tag)')
    
    # Kritikalität basierend auf Kapazität
    df['criticality'] = pd.cut(
        df['capacity_max'].fillna(0), 
        bins=[0, 5000, 15000, 50000, float('inf')],
        labels=['Niedrig', 'Mittel', 'Hoch', 'Kritisch']
    ).astype(str)
    
    # Asset-Namen generieren
    df['asset_name'] = df.apply(lambda row: generate_asset_name(row), axis=1)
    
    print(f"🔧 Feature Engineering...")
    
    # ERWEITERTE FEATURES ableiten
    
    # 1. Energieeffizienz-Features
    df['energy_efficiency'] = df['motor_power_kw'] / (df['total_power_consumption'] + 0.1)
    df['power_density'] = df['total_power_consumption'] / (df['capacity_max'] + 1)
    
    # 2. Wasser-Effizienz-Features  
    df['water_efficiency'] = df['water_consumption_ls'] / (df['capacity_max'] + 1)
    df['water_per_liter_capacity'] = df['water_consumption_ls'] / (df['capacity_min'] + 1)
    
    # 3. Mechanische Komplexität
    complexity_scores = {
        'integrated direct drive': 1,    # Einfach
        'flat - belt drive': 2,          # Mittel
        'gear drive': 3                  # Komplex
    }
    df['complexity_score'] = df['drive_type'].map(complexity_scores).fillna(2)
    
    # 4. Qualitäts-Features
    quality_scores = {
        'premium - Level': 2,
        'standard - Level': 1
    }
    df['quality_score'] = df['quality_level'].map(quality_scores).fillna(1)
    
    # 5. Größen-Features
    df['volume_m3'] = (df['length_mm'] * df['width_mm'] * df['height_mm']) / 1e9
    df['weight_per_volume'] = df['total_weight_kg'] / (df['volume_m3'] + 0.1)
    
    print(f"💰 Berechne erweiterte Betriebskosten...")
    
    # ERWEITERTE BETRIEBSKOSTEN berechnen
    annual_costs = []
    
    for idx, row in df.iterrows():
        costs = calculate_extended_operating_costs(row)
        annual_costs.append(costs['total_annual_cost'])
        
        # Einzelkomponenten auch speichern
        for component, cost in costs.items():
            if component != 'total_annual_cost':
                df.loc[idx, f'annual_{component}'] = cost
    
    df['annual_maintenance'] = annual_costs
    
    # Wartungsratio berechnen
    df['maintenance_ratio'] = df['annual_maintenance'] / (df['purchase_price'] + 1)
    
    print(f"✅ Feature Engineering abgeschlossen")
    print(f"📊 Durchschnittliche jährliche Betriebskosten: €{df['annual_maintenance'].mean():,.0f}")
    print(f"📊 Bereich: €{df['annual_maintenance'].min():,.0f} - €{df['annual_maintenance'].max():,.0f}")
    
    return df

def generate_asset_name(row):
    """Generiert realistische Asset-Namen für Zentrifugen"""
    
    category_codes = {
        'Citrus': 'CIT',
        'Wine': 'WIN', 
        'Dairy': 'DAI',
        'Industrial': 'IND'
    }
    
    code = category_codes.get(row['category'], 'SEP')
    model_code = str(row['model']).split('-')[0] if pd.notna(row['model']) else 'GFA'
    
    return f"{code}-{model_code}-{np.random.randint(100, 999)}"

def calculate_extended_operating_costs(asset_row) -> dict:
    """
    Berechnet erweiterte Betriebskosten für einzelne Zentrifuge
    
    Args:
        asset_row: Pandas Series mit Asset-Daten
    
    Returns:
        Dictionary mit Kostenkomponenten
    """
    
    # Base Maintenance (traditionell)
    purchase_price = asset_row.get('purchase_price', 100000)
    base_maintenance_rate = 0.12  # 12% für Zentrifugen
    
    # Modifikationen basierend auf Features
    quality_factor = 1.2 if asset_row.get('quality_level') == 'premium - Level' else 1.0
    complexity_factor = asset_row.get('complexity_score', 2) * 0.15 + 0.7  # 0.85 - 1.15
    
    base_maintenance = purchase_price * base_maintenance_rate * quality_factor * complexity_factor
    
    # ENERGIEKOSTEN
    power_consumption = asset_row.get('total_power_consumption', 20)
    usage_pattern = asset_row.get('usage_pattern', 'Standard (8h/Tag)')
    location = asset_row.get('location', 'Düsseldorf (HQ)')
    
    # Betriebsstunden pro Jahr
    annual_hours = {
        "Gelegentlich": 1000,
        "Standard (8h/Tag)": 2000,
        "Extended (12h/Tag)": 3500,  # Lebensmittel-Saison
        "24/7 Betrieb": 8000  # Wartungspausen eingerechnet
    }.get(usage_pattern, 2000)
    
    # Regionale Strompreise (Industriestrom)
    electricity_prices = {
        'Düsseldorf (HQ)': 0.28,    # Deutschland hoch
        'Oelde': 0.26,              # Deutschland regional  
        'Berlin': 0.27,
        'Hamburg': 0.28,
        'Kopenhagen': 0.32,         # Dänemark sehr hoch
        'Mailand': 0.25,            # Italien mittel
        'Shanghai': 0.08,           # China niedrig
        'Singapur': 0.18,           # Asien mittel
        'Chicago': 0.12,            # USA niedrig
        'São Paulo': 0.15           # Brasilien mittel
    }
    
    electricity_price = electricity_prices.get(location, 0.26)
    energy_cost = power_consumption * annual_hours * electricity_price
    
    # WASSERKOSTEN
    water_consumption = asset_row.get('water_consumption_ls', 0.5)
    water_per_ejection = asset_row.get('water_per_ejection', 2)
    
    # Regionale Wasserpreise (Industriewasser)
    water_prices = {
        'Düsseldorf (HQ)': 0.0025,
        'Oelde': 0.002,
        'Berlin': 0.0028,
        'Hamburg': 0.0024,
        'Kopenhagen': 0.0035,   # Dänemark teuer
        'Mailand': 0.002,       # Italien günstiger
        'Shanghai': 0.0008,     # China sehr günstig
        'Singapur': 0.003,      # Wassermangel
        'Chicago': 0.0015,
        'São Paulo': 0.001
    }
    
    water_price = water_prices.get(location, 0.002)
    
    # Wasserverbrauch pro Stunde (Betrieb + Reinigung)
    hourly_water = water_consumption + (water_per_ejection * 2)  # 2 Ejections/h angenommen
    water_cost = hourly_water * annual_hours * water_price
    
    # PERSONALKOSTEN
    # Komplexere Maschinen brauchen mehr qualifiziertes Personal
    base_operator_hours = {
        'premium - Level': 200,  # Hohe Automatisierung
        'standard - Level': 350  # Mehr manuelle Eingriffe
    }.get(asset_row.get('quality_level', 'standard - Level'), 350)
    
    complexity_multiplier = asset_row.get('complexity_score', 2) * 0.2 + 0.6  # 0.8 - 1.2
    
    # Regionale Lohnkosten (Maschinenbediener/Techniker)
    hourly_wages = {
        'Düsseldorf (HQ)': 48,    # Deutschland hoch
        'Oelde': 42,              # Deutschland regional
        'Berlin': 45,
        'Hamburg': 47,
        'Kopenhagen': 58,         # Dänemark sehr hoch
        'Mailand': 38,            # Italien mittel
        'Shanghai': 12,           # China niedrig
        'Singapur': 25,           # Asien entwickelt
        'Chicago': 35,            # USA mittel
        'São Paulo': 15           # Brasilien niedrig
    }
    
    wage_per_hour = hourly_wages.get(location, 42)
    operator_hours = base_operator_hours * complexity_multiplier
    personnel_cost = operator_hours * wage_per_hour
    
    # ZUSÄTZLICHE ZENTRIFUGEN-SPEZIFISCHE KOSTEN
    
    # Ersatzteil-Verfügbarkeit (Premium = bessere Verfügbarkeit)
    spare_parts_factor = 0.8 if asset_row.get('quality_level') == 'premium - Level' else 1.0
    spare_parts_cost = base_maintenance * 0.3 * spare_parts_factor
    
    # Reinigungskosten (besonders wichtig bei Lebensmitteln)
    cleaning_cost = 0
    if asset_row.get('category') in ['Citrus', 'Wine', 'Dairy']:
        # CIP-Reinigung, Chemikalien, Zeit
        cleaning_cost = purchase_price * 0.02  # 2% für Lebensmittel-Hygiene
    
    # Vibrationsüberwachung (bei kritischen Assets)
    monitoring_cost = 0
    if asset_row.get('criticality') in ['Hoch', 'Kritisch']:
        monitoring_cost = 2500  # IoT-Sensoren, Cloud-Service
    
    total_cost = (base_maintenance + energy_cost + water_cost + personnel_cost + 
                  spare_parts_cost + cleaning_cost + monitoring_cost)
    
    return {
        'base_maintenance': base_maintenance,
        'energy_cost': energy_cost,
        'water_cost': water_cost,
        'personnel_cost': personnel_cost,
        'spare_parts_cost': spare_parts_cost,
        'cleaning_cost': cleaning_cost,
        'monitoring_cost': monitoring_cost,
        'total_annual_cost': total_cost
    }

def create_mock_centrifuge_data() -> pd.DataFrame:
    """Erstellt Mock-Daten falls Excel nicht verfügbar"""
    
    print("⚠️ Erstelle Mock-Zentrifugen-Daten")
    
    mock_data = []
    
    # Realistische GEA-Zentrifugen
    centrifuge_types = [
        {'category': 'Citrus', 'subcategory': 'Citrus Juice Clarification', 'model': 'GFA 200-30-820', 
         'price': 344261, 'motor_power': 55, 'total_power': 44, 'water': 1.2},
        {'category': 'Citrus', 'subcategory': 'Citrus Juice Clarification', 'model': 'GFA 100-69-357',
         'price': 234070, 'motor_power': 45, 'total_power': 36, 'water': 0.8},
        {'category': 'Wine', 'subcategory': 'Clarific. of Sparkling Wine', 'model': 'GFA 10-43-210',
         'price': 83397, 'motor_power': 7.5, 'total_power': 6, 'water': 0.5},
        {'category': 'Dairy', 'subcategory': 'Milk Clarification', 'model': 'MSE 300-01-777',
         'price': 256000, 'motor_power': 37, 'total_power': 30, 'water': 0.9}
    ]
    
    for i, base_type in enumerate(centrifuge_types * 3):  # 12 Assets
        asset = base_type.copy()
        asset['asset_name'] = f"SEP-{asset['model'].split('-')[0]}-{i+1:03d}"
        asset['manufacturer'] = 'GEA'
        asset['purchase_price'] = asset['price']
        asset['motor_power_kw'] = asset['motor_power']
        asset['total_power_consumption'] = asset['total_power']
        asset['water_consumption_ls'] = asset['water']
        asset['age_years'] = np.random.uniform(0.5, 8)
        asset['warranty_years'] = 2
        asset['expected_lifetime'] = 15
        asset['location'] = np.random.choice(['Düsseldorf (HQ)', 'Oelde', 'Berlin'])
        asset['usage_pattern'] = 'Extended (12h/Tag)'
        asset['criticality'] = 'Hoch'
        asset['drive_type'] = np.random.choice(['integrated direct drive', 'flat - belt drive'])
        asset['quality_level'] = np.random.choice(['premium - Level', 'standard - Level'])
        
        mock_data.append(asset)
    
    df = pd.DataFrame(mock_data)
    
    # Features berechnen
    df['energy_efficiency'] = df['motor_power_kw'] / df['total_power_consumption']
    df['complexity_score'] = 2
    df['quality_score'] = 1
    
    # Betriebskosten berechnen
    annual_costs = []
    for _, row in df.iterrows():
        costs = calculate_extended_operating_costs(row)
        annual_costs.append(costs['total_annual_cost'])
    
    df['annual_maintenance'] = annual_costs
    df['maintenance_ratio'] = df['annual_maintenance'] / df['purchase_price']
    
    return df

if __name__ == "__main__":
    # Test des Data Loaders
    print("🧪 Teste Zentrifugen Data Loader...\n")
    
    # Versuche echte Excel-Datei zu laden
    excel_path = "HinterlandHack _ FinaleListe.xlsx"
    
    df = load_centrifuge_data(excel_path)
    
    print(f"\n📊 Geladene Daten:")
    print(f"   • {len(df)} Zentrifugen")
    print(f"   • {len(df.columns)} Features")
    print(f"   • Durchschnittlicher Listenpreis: €{df['purchase_price'].mean():,.0f}")
    print(f"   • Durchschnittliche Betriebskosten: €{df['annual_maintenance'].mean():,.0f}")
    
    print(f"\n🔧 Wichtigste Features:")
    key_features = ['purchase_price', 'motor_power_kw', 'total_power_consumption', 
                   'water_consumption_ls', 'annual_maintenance']
    print(df[key_features].describe())
    
    # Speichern für ML-Training
    df.to_csv('data/centrifuge_training_data.csv', index=False)
    print(f"\n💾 Daten gespeichert: data/centrifuge_training_data.csv")