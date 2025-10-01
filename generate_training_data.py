import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class AssetTemplate:
    """Template für verschiedene Asset-Typen"""
    category: str
    subcategory: str
    price_range: tuple  # (min, max)
    base_maintenance_rate: float  # % of purchase price
    common_manufacturers: List[str]
    typical_lifetime: int  # years
    usage_patterns: List[str]

def get_asset_templates() -> List[AssetTemplate]:
    """Definiert realistische Asset-Templates"""
    
    templates = [
        # IT-Equipment
        AssetTemplate(
            category="IT-Equipment", subcategory="Server",
            price_range=(5000, 25000), base_maintenance_rate=0.18,
            common_manufacturers=["Dell", "HP", "Lenovo", "IBM"],
            typical_lifetime=5, usage_patterns=["24/7 Betrieb", "Extended (12h/Tag)"]
        ),
        AssetTemplate(
            category="IT-Equipment", subcategory="Laptop", 
            price_range=(800, 3500), base_maintenance_rate=0.12,
            common_manufacturers=["Dell", "HP", "Lenovo", "Apple"],
            typical_lifetime=4, usage_patterns=["Standard (8h/Tag)", "Extended (12h/Tag)"]
        ),
        AssetTemplate(
            category="IT-Equipment", subcategory="Workstation",
            price_range=(2000, 8000), base_maintenance_rate=0.15,
            common_manufacturers=["Dell", "HP", "Lenovo"],
            typical_lifetime=5, usage_patterns=["Standard (8h/Tag)", "Extended (12h/Tag)"]
        ),
        AssetTemplate(
            category="IT-Equipment", subcategory="Netzwerk",
            price_range=(1000, 15000), base_maintenance_rate=0.10,
            common_manufacturers=["Cisco", "HP", "Dell", "Netgear"],
            typical_lifetime=7, usage_patterns=["24/7 Betrieb"]
        ),
        
        # Industrial Equipment
        AssetTemplate(
            category="Industrial", subcategory="Separator",
            price_range=(80000, 300000), base_maintenance_rate=0.14,
            common_manufacturers=["GEA", "Alfa Laval", "Flottweg"],
            typical_lifetime=15, usage_patterns=["24/7 Betrieb", "Standard (8h/Tag)"]
        ),
        AssetTemplate(
            category="Industrial", subcategory="Homogenizer",
            price_range=(60000, 200000), base_maintenance_rate=0.16,
            common_manufacturers=["GEA", "Tetra Pak", "APV"],
            typical_lifetime=12, usage_patterns=["Extended (12h/Tag)", "24/7 Betrieb"]
        ),
        AssetTemplate(
            category="Industrial", subcategory="Pump",
            price_range=(5000, 80000), base_maintenance_rate=0.12,
            common_manufacturers=["GEA", "Grundfos", "KSB", "Alfa Laval"],
            typical_lifetime=10, usage_patterns=["24/7 Betrieb", "Extended (12h/Tag)"]
        ),
        AssetTemplate(
            category="Industrial", subcategory="Pasteurizer",
            price_range=(100000, 500000), base_maintenance_rate=0.13,
            common_manufacturers=["GEA", "Tetra Pak", "SPX"],
            typical_lifetime=20, usage_patterns=["Extended (12h/Tag)", "24/7 Betrieb"]
        ),
        
        # Software
        AssetTemplate(
            category="Software", subcategory="ERP",
            price_range=(50000, 500000), base_maintenance_rate=0.20,
            common_manufacturers=["SAP", "Oracle", "Microsoft"],
            typical_lifetime=7, usage_patterns=["24/7 Betrieb"]
        ),
        AssetTemplate(
            category="Software", subcategory="CAD",
            price_range=(5000, 50000), base_maintenance_rate=0.18,
            common_manufacturers=["Autodesk", "Siemens", "SolidWorks"],
            typical_lifetime=5, usage_patterns=["Standard (8h/Tag)"]
        ),
        
        # Vehicles
        AssetTemplate(
            category="Vehicles", subcategory="PKW",
            price_range=(25000, 80000), base_maintenance_rate=0.08,
            common_manufacturers=["BMW", "Mercedes", "VW", "Audi"],
            typical_lifetime=8, usage_patterns=["Standard (8h/Tag)", "Gelegentlich"]
        ),
        AssetTemplate(
            category="Vehicles", subcategory="LKW",
            price_range=(80000, 200000), base_maintenance_rate=0.12,
            common_manufacturers=["MAN", "Mercedes", "Volvo", "Scania"],
            typical_lifetime=10, usage_patterns=["Extended (12h/Tag)", "24/7 Betrieb"]
        )
    ]
    
    return templates

def get_realistic_locations() -> List[Dict]:
    """GEA Standorte mit regionalen Faktoren"""
    return [
        {"name": "Düsseldorf (HQ)", "country": "DE", "cost_factor": 0.95, "weight": 25},
        {"name": "Oelde", "country": "DE", "cost_factor": 1.00, "weight": 20},
        {"name": "Berlin", "country": "DE", "cost_factor": 1.05, "weight": 15},
        {"name": "Hamburg", "country": "DE", "cost_factor": 1.03, "weight": 10},
        {"name": "München", "country": "DE", "cost_factor": 1.08, "weight": 8},
        {"name": "Kopenhagen", "country": "DK", "cost_factor": 1.15, "weight": 5},
        {"name": "Mailand", "country": "IT", "cost_factor": 1.12, "weight": 4},
        {"name": "Lyon", "country": "FR", "cost_factor": 1.10, "weight": 3},
        {"name": "Shanghai", "country": "CN", "cost_factor": 0.85, "weight": 5},
        {"name": "Singapur", "country": "SG", "cost_factor": 1.20, "weight": 2},
        {"name": "Chicago", "country": "US", "cost_factor": 1.25, "weight": 2},
        {"name": "São Paulo", "country": "BR", "cost_factor": 0.75, "weight": 1}
    ]

def calculate_realistic_maintenance(asset: Dict, template: AssetTemplate) -> float:
    """Berechnet realistische Wartungskosten mit vielen Faktoren"""
    
    base_cost = asset['purchase_price'] * template.base_maintenance_rate
    
    # Manufacturer factors (Premium vs Budget)
    manufacturer_factors = {
        # Premium brands
        "Dell": 1.05, "Siemens": 1.15, "GEA": 1.10, "SAP": 1.20, "BMW": 1.15,
        "Cisco": 1.10, "Autodesk": 1.05, "Mercedes": 1.20,
        
        # Standard brands
        "HP": 1.00, "Lenovo": 0.95, "Alfa Laval": 1.00, "Microsoft": 1.00,
        "Oracle": 1.10, "VW": 0.90, "MAN": 1.00,
        
        # Budget/Regional brands
        "Netgear": 0.85, "Grundfos": 0.95, "KSB": 0.90, "Volvo": 1.05
    }
    
    mfg_factor = manufacturer_factors.get(asset['manufacturer'], 1.0)
    
    # Age factor (exponential increase)
    age_factor = 1.0 + (asset['age_years'] * 0.1) + (asset['age_years'] ** 1.5 * 0.02)
    
    # Usage intensity factor
    usage_factors = {
        "Gelegentlich": 0.70,
        "Standard (8h/Tag)": 1.00,
        "Extended (12h/Tag)": 1.25,
        "24/7 Betrieb": 1.80
    }
    usage_factor = usage_factors.get(asset['usage_pattern'], 1.0)
    
    # Criticality factor
    criticality_factors = {
        "Niedrig": 0.80, "Mittel": 1.00, "Hoch": 1.30, "Kritisch": 1.60
    }
    crit_factor = criticality_factors.get(asset['criticality'], 1.0)
    
    # Location factor
    locations = get_realistic_locations()
    location_data = next((l for l in locations if l['name'] == asset['location']), 
                        {"cost_factor": 1.0})
    location_factor = location_data['cost_factor']
    
    # Warranty factor (less maintenance in warranty period)
    warranty_factor = 0.7 if asset['age_years'] < asset['warranty_years'] else 1.0
    
    # Calculate final maintenance cost
    maintenance_cost = (base_cost * mfg_factor * age_factor * usage_factor * 
                       crit_factor * location_factor * warranty_factor)
    
    # Add realistic variance (some assets are just unlucky)
    variance = np.random.normal(1.0, 0.15)  # ±15% standard deviation
    final_cost = maintenance_cost * max(0.3, variance)  # Minimum 30% of expected
    
    return round(final_cost, 2)

def generate_asset_name(category: str, subcategory: str, location: str, index: int) -> str:
    """Generiert realistische Asset-Namen"""
    
    # Location codes
    location_codes = {
        "Düsseldorf (HQ)": "DUS", "Oelde": "OEL", "Berlin": "BER",
        "Hamburg": "HH", "München": "MUC", "Kopenhagen": "CPH",
        "Mailand": "MIL", "Lyon": "LYO", "Shanghai": "SHA",
        "Singapur": "SIN", "Chicago": "CHI", "São Paulo": "SAO"
    }
    
    loc_code = location_codes.get(location, "XXX")
    
    # Category prefixes
    if category == "IT-Equipment":
        if subcategory == "Server":
            return f"SRV-{loc_code}-{index:03d}"
        elif subcategory == "Laptop":
            return f"LAP-{loc_code}-{index:03d}"
        elif subcategory == "Workstation":
            return f"WS-{loc_code}-{index:03d}"
        else:
            return f"NET-{loc_code}-{index:03d}"
    elif category == "Industrial":
        prefixes = {"Separator": "SEP", "Homogenizer": "HOM", "Pump": "PMP", "Pasteurizer": "PST"}
        prefix = prefixes.get(subcategory, "IND")
        return f"{prefix}-{loc_code}-{index:03d}"
    elif category == "Software":
        return f"SW-{subcategory[:3].upper()}-{index:03d}"
    elif category == "Vehicles":
        prefixes = {"PKW": "CAR", "LKW": "TRK"}
        prefix = prefixes.get(subcategory, "VEH")
        return f"{prefix}-{loc_code}-{index:03d}"
    
    return f"AST-{loc_code}-{index:03d}"

def generate_realistic_dataset(num_assets: int = 500) -> pd.DataFrame:
    """Generiert kompletten realistischen Dataset"""
    
    templates = get_asset_templates()
    locations = get_realistic_locations()
    
    # Create weighted location choices
    location_names = [loc['name'] for loc in locations]
    location_weights = [loc['weight'] for loc in locations]
    
    assets = []
    
    print(f"🏭 Generiere {num_assets} realistische Assets...")
    
    for i in range(num_assets):
        # Choose asset template (weighted by realism)
        template_weights = [10, 8, 5, 3, 4, 3, 6, 2, 3, 4, 6, 4]  # More IT & Industrial
        template = np.random.choice(templates, p=np.array(template_weights)/sum(template_weights))
        
        # Generate basic asset info
        purchase_price = round(np.random.uniform(*template.price_range), 2)
        manufacturer = np.random.choice(template.common_manufacturers)
        location = np.random.choice(location_names, p=np.array(location_weights)/sum(location_weights))
        
        # Purchase date (last 5 years, weighted towards recent)
        days_ago = np.random.exponential(365)  # Exponential distribution
        days_ago = min(days_ago, 5*365)  # Cap at 5 years
        purchase_date = datetime.now() - timedelta(days=int(days_ago))
        age_years = (datetime.now() - purchase_date).days / 365.25
        
        # Other realistic attributes
        usage_pattern = np.random.choice(template.usage_patterns)
        criticality = np.random.choice(["Niedrig", "Mittel", "Hoch", "Kritisch"], 
                                     p=[0.2, 0.5, 0.25, 0.05])
        warranty_years = np.random.choice([1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        
        # Generate asset
        asset = {
            'asset_id': f"A{i+1:04d}",
            'asset_name': generate_asset_name(template.category, template.subcategory, location, i+1),
            'category': template.category,
            'subcategory': template.subcategory,
            'manufacturer': manufacturer,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date.date(),
            'age_years': round(age_years, 2),
            'location': location,
            'usage_pattern': usage_pattern,
            'criticality': criticality,
            'warranty_years': warranty_years,
            'expected_lifetime': template.typical_lifetime
        }
        
        # Calculate realistic maintenance cost
        annual_maintenance = calculate_realistic_maintenance(asset, template)
        asset['annual_maintenance'] = annual_maintenance
        asset['maintenance_ratio'] = round(annual_maintenance / purchase_price, 4)
        
        assets.append(asset)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"   ✅ {i+1}/{num_assets} Assets generiert...")
    
    df = pd.DataFrame(assets)
    
    print(f"🎉 Dataset komplett! {len(df)} Assets mit realistischen Mustern erstellt.")
    print(f"📊 Durchschnittliche Wartungskosten: €{df['annual_maintenance'].mean():,.0f}")
    print(f"📊 Wartungsratio-Bereich: {df['maintenance_ratio'].min():.1%} - {df['maintenance_ratio'].max():.1%}")
    
    return df

def add_data_quality_issues(df: pd.DataFrame, missing_rate: float = 0.1) -> pd.DataFrame:
    """Fügt realistische Datenqualitätsprobleme hinzu"""
    
    df_messy = df.copy()
    n_rows = len(df_messy)
    
    print(f"🔧 Füge realistische Datenqualitätsprobleme hinzu...")
    
    # Missing values in verschiedenen Spalten
    missing_columns = ['manufacturer', 'warranty_years', 'usage_pattern']
    for col in missing_columns:
        n_missing = int(n_rows * missing_rate * np.random.uniform(0.5, 1.5))
        missing_indices = np.random.choice(df_messy.index, n_missing, replace=False)
        df_messy.loc[missing_indices, col] = np.nan
    
    # Inconsistent manufacturer names
    df_messy.loc[df_messy['manufacturer'] == 'Dell', 'manufacturer'] = \
        np.random.choice(['Dell', 'DELL', 'Dell Inc.'], 
                        size=sum(df_messy['manufacturer'] == 'Dell'),
                        p=[0.7, 0.2, 0.1])
    
    # Some unrealistic outliers (data entry errors)
    n_outliers = int(n_rows * 0.02)  # 2% outliers
    outlier_indices = np.random.choice(df_messy.index, n_outliers, replace=False)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            # Unrealistically high maintenance
            df_messy.loc[idx, 'annual_maintenance'] *= np.random.uniform(3, 10)
        else:
            # Unrealistically low maintenance
            df_messy.loc[idx, 'annual_maintenance'] *= np.random.uniform(0.1, 0.3)
    
    print(f"   ✅ {missing_rate*100:.0f}% Missing Values hinzugefügt")
    print(f"   ✅ {n_outliers} Outliers hinzugefügt")
    
    return df_messy

if __name__ == "__main__":
    # Generate dataset
    print("🚀 Starte Generierung des ML-Training-Datasets...\n")
    
    # Create clean dataset
    df_clean = generate_realistic_dataset(num_assets=500)
    
    # Add realistic data quality issues
    df_realistic = add_data_quality_issues(df_clean, missing_rate=0.08)
    
    # Save datasets
    df_clean.to_csv('data/training_data_clean.csv', index=False)
    df_realistic.to_csv('data/training_data_realistic.csv', index=False)
    
    print(f"\n💾 Datasets gespeichert:")
    print(f"   📄 data/training_data_clean.csv ({len(df_clean)} Assets)")
    print(f"   📄 data/training_data_realistic.csv ({len(df_realistic)} Assets)")
    
    # Show sample
    print(f"\n📋 Sample der generierten Daten:")
    print(df_realistic[['asset_name', 'category', 'manufacturer', 'purchase_price', 
                       'annual_maintenance', 'maintenance_ratio']].head())
    
    print(f"\n🎯 Ready für ML-Training! 🤖")