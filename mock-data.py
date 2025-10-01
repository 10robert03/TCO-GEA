import pandas as pd
import random
from datetime import datetime, timedelta

def get_mock_assets():
    """Erstellt realistische Demo-Assets für GEA"""
    
    assets = [
        # IT-Equipment
        {"id": "IT-001", "name": "SRV-DUS-001", "category": "Server", "manufacturer": "Dell", 
         "model": "PowerEdge R740", "price": 8500, "annual_maintenance": 1700, "location": "Düsseldorf", "age": 2},
        
        {"id": "IT-002", "name": "SRV-DUS-002", "category": "Server", "manufacturer": "HP", 
         "model": "ProLiant DL380", "price": 7200, "annual_maintenance": 1440, "location": "Düsseldorf", "age": 1},
        
        {"id": "IT-003", "name": "WS-BER-015", "category": "Workstation", "manufacturer": "Dell", 
         "model": "Precision 7760", "price": 4500, "annual_maintenance": 675, "location": "Berlin", "age": 1},
        
        {"id": "IT-004", "name": "LAP-HH-089", "category": "Laptop", "manufacturer": "Lenovo", 
         "model": "ThinkPad X1 Carbon", "price": 2200, "annual_maintenance": 330, "location": "Hamburg", "age": 3},
        
        # Industrieanlagen
        {"id": "PRD-001", "name": "Separator-A12", "category": "Separator", "manufacturer": "GEA", 
         "model": "WSP 5000", "price": 125000, "annual_maintenance": 18750, "location": "Oelde", "age": 3},
        
        {"id": "PRD-002", "name": "Homogenizer-B05", "category": "Homogenizer", "manufacturer": "GEA", 
         "model": "Ariete 5400", "price": 95000, "annual_maintenance": 14250, "location": "Oelde", "age": 2},
        
        {"id": "PRD-003", "name": "Pump-C18", "category": "Pump", "manufacturer": "Alfa Laval", 
         "model": "LKH Prime", "price": 35000, "annual_maintenance": 5250, "location": "Düsseldorf", "age": 4},
        
        # Software
        {"id": "SW-001", "name": "SAP-ERP-Main", "category": "Software", "manufacturer": "SAP", 
         "model": "S/4HANA", "price": 450000, "annual_maintenance": 90000, "location": "Düsseldorf", "age": 1},
        
        {"id": "SW-002", "name": "AutoCAD-Licenses", "category": "Software", "manufacturer": "Autodesk", 
         "model": "AutoCAD Plant 3D", "price": 25000, "annual_maintenance": 6250, "location": "Berlin", "age": 2},
    ]
    
    return pd.DataFrame(assets)

def get_dashboard_metrics():
    """KPIs für Dashboard"""
    return {
        "total_assets": 1247,
        "total_tco": 12400000,  # €12.4M
        "estimated_costs": 2100000,  # €2.1M
        "estimated_percentage": 17
    }

def get_manufacturers():
    """Liste der verfügbaren Hersteller"""
    return {
        "IT-Equipment": ["Dell", "HP", "Lenovo", "Apple", "Microsoft"],
        "Industrial": ["GEA", "Alfa Laval", "Siemens", "ABB", "Schneider Electric"],
        "Software": ["SAP", "Microsoft", "Autodesk", "Oracle", "Adobe"]
    }

def get_asset_categories():
    """Asset-Kategorien mit Icons"""
    return {
        "IT-Equipment": {
            "icon": "💻",
            "description": "Server, Laptops, Netzwerk",
            "subcategories": ["Server", "Laptop", "Workstation", "Netzwerk"]
        },
        "Industrial": {
            "icon": "🏭", 
            "description": "Maschinen & Anlagen",
            "subcategories": ["Separator", "Homogenizer", "Pump", "Pasteurizer"]
        },
        "Software": {
            "icon": "💾",
            "description": "Lizenzen & SaaS",
            "subcategories": ["ERP", "CAD", "Office", "Analyse"]
        },
        "Vehicles": {
            "icon": "🚗",
            "description": "Fuhrpark & Transport", 
            "subcategories": ["PKW", "LKW", "Gabelstapler"]
        },
        "Facilities": {
            "icon": "🏢",
            "description": "Gebäude & Infrastruktur",
            "subcategories": ["HVAC", "Security", "Cleaning"]
        }
    }

def get_similar_assets(category, manufacturer, price_range=0.3):
    """Findet ähnliche Assets für Referenz"""
    all_assets = get_mock_assets()
    
    # Filter nach Kategorie und Hersteller
    similar = all_assets[
        (all_assets['category'] == category) | 
        (all_assets['manufacturer'] == manufacturer)
    ]
    
    return similar.head(3)  # Top 3 ähnliche Assets

def calculate_fake_tco_prediction(asset_type, manufacturer, price):
    """Simuliert ML-Vorhersage für Demo"""
    
    # Base maintenance rates by category
    base_rates = {
        "Server": 0.20,
        "Laptop": 0.15, 
        "Workstation": 0.18,
        "Separator": 0.15,
        "Homogenizer": 0.15,
        "Pump": 0.15,
        "Software": 0.20
    }
    
    # Manufacturer factors
    manufacturer_factors = {
        "Dell": 1.1, "HP": 1.0, "Lenovo": 0.95,
        "GEA": 1.15, "Alfa Laval": 1.05, "Siemens": 1.2,
        "SAP": 1.3, "Microsoft": 1.0, "Autodesk": 1.1
    }
    
    base_rate = base_rates.get(asset_type, 0.15)
    mfg_factor = manufacturer_factors.get(manufacturer, 1.0)
    
    # Berechnung mit etwas Varianz
    annual_maintenance = price * base_rate * mfg_factor
    variance = random.uniform(0.8, 1.2)  # ±20% Varianz
    
    predicted_cost = annual_maintenance * variance
    confidence = random.uniform(0.75, 0.95)  # 75-95% Konfidenz
    
    # Confidence-Level bestimmen
    if confidence > 0.85:
        confidence_level = "Hoch"
        confidence_color = "confidence-high"
    elif confidence > 0.70:
        confidence_level = "Mittel"
        confidence_color = "confidence-medium"
    else:
        confidence_level = "Niedrig"
        confidence_color = "confidence-low"
    
    return {
        "prediction": round(predicted_cost),
        "confidence": round(confidence * 100),
        "confidence_level": confidence_level,
        "confidence_color": confidence_color,
        "range_min": round(predicted_cost * 0.8),
        "range_max": round(predicted_cost * 1.2)
    }