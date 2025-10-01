"""
Erweiterbares TCO-Komponenten-System für präzise Kostenberechnung
Berücksichtigt Energie, Wasser, Personal und weitere Faktoren
Mit Energy Agent Integration für Echtzeit-Strompreise
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

@dataclass
class TCOComponent:
    """Einzelne TCO-Komponente mit detaillierten Informationen"""
    name: str
    annual_cost: float
    category: str  # 'fixed', 'variable', 'one_time'
    confidence: float  # 0.0 - 1.0
    calculation_method: str
    factors: Dict[str, Any]
    region_dependent: bool = True
    equipment_dependent: bool = True

class ExtendedTCOCalculator:
    """
    Erweiterbarer TCO-Rechner mit modularen Komponenten
    Speziell optimiert für industrielle Zentrifugen
    Mit Energy Agent Integration für Echtzeit-Strompreise
    """
    
    def __init__(self):
        self.components = {}
        self.regional_factors = self._init_regional_factors()
        self.industry_standards = self._init_industry_standards()
        self.calculation_history = []
    
    def _init_regional_factors(self) -> Dict[str, Dict[str, float]]:
        """Regionale Kostenfaktoren für verschiedene TCO-Komponenten"""
        return {
            'electricity_prices': {  # €/kWh Industriestrom
                'Düsseldorf (HQ)': 0.28,
                'Oelde': 0.26,
                'Berlin': 0.27,
                'Hamburg': 0.28,
                'München': 0.29,
                'Kopenhagen': 0.32,
                'Mailand': 0.25,
                'Lyon': 0.24,
                'Shanghai': 0.08,
                'Singapur': 0.18,
                'Chicago': 0.12,
                'São Paulo': 0.15,
                'Andere': 0.25
            },
            'water_prices': {  # €/Liter Industriewasser
                'Düsseldorf (HQ)': 0.0025,
                'Oelde': 0.002,
                'Berlin': 0.0028,
                'Hamburg': 0.0024,
                'München': 0.003,
                'Kopenhagen': 0.0035,
                'Mailand': 0.002,
                'Lyon': 0.0022,
                'Shanghai': 0.0008,
                'Singapur': 0.003,
                'Chicago': 0.0015,
                'São Paulo': 0.001,
                'Andere': 0.002
            },
            'labor_costs': {  # €/Stunde Maschinenbediener/Techniker
                'Düsseldorf (HQ)': 48,
                'Oelde': 42,
                'Berlin': 45,
                'Hamburg': 47,
                'München': 50,
                'Kopenhagen': 58,
                'Mailand': 38,
                'Lyon': 35,
                'Shanghai': 12,
                'Singapur': 25,
                'Chicago': 35,
                'São Paulo': 15,
                'Andere': 40
            },
            'regulatory_compliance': {  # Relative Faktoren für Compliance-Kosten
                'Düsseldorf (HQ)': 1.2,  # Deutschland: strenge Regeln
                'Oelde': 1.2,
                'Berlin': 1.2,
                'Hamburg': 1.2,
                'München': 1.2,
                'Kopenhagen': 1.3,      # EU + nationale Standards
                'Mailand': 1.1,         # EU Standards
                'Lyon': 1.1,
                'Shanghai': 0.8,        # Weniger strenge Standards
                'Singapur': 1.0,
                'Chicago': 0.9,         # USA Standards
                'São Paulo': 0.7,       # Entwicklungsland
                'Andere': 1.0
            }
        }
    
    def _init_industry_standards(self) -> Dict[str, Any]:
        """Industrie-Standards und Benchmark-Werte"""
        return {
            'centrifuge_base_maintenance': {
                'disc_stack': 0.12,      # 12% vom Anschaffungspreis
                'decanter': 0.14,        # 14% vom Anschaffungspreis  
                'chamber_bowl': 0.10     # 10% vom Anschaffungspreis
            },
            'annual_operating_hours': {
                'Gelegentlich': 1000,
                'Standard (8h/Tag)': 2000,
                'Extended (12h/Tag)': 3500,  # Saisonale Spitzen
                '24/7 Betrieb': 8000         # Abzüglich Wartung/Störungen
            },
            'complexity_factors': {
                'integrated direct drive': {'maintenance': 0.85, 'personnel': 0.7},
                'flat - belt drive': {'maintenance': 1.0, 'personnel': 1.0},
                'gear drive': {'maintenance': 1.15, 'personnel': 1.3}
            },
            'quality_factors': {
                'premium - Level': {'maintenance': 1.2, 'reliability': 0.95, 'personnel': 0.6},
                'standard - Level': {'maintenance': 1.0, 'reliability': 0.85, 'personnel': 1.0}
            },
            'criticality_factors': {
                'Niedrig': {'downtime_cost': 0.02, 'monitoring': 0},
                'Mittel': {'downtime_cost': 0.05, 'monitoring': 1000},
                'Hoch': {'downtime_cost': 0.10, 'monitoring': 2500},
                'Kritisch': {'downtime_cost': 0.20, 'monitoring': 5000}
            }
        }
    
    def add_base_maintenance_component(self, asset_data: Dict) -> TCOComponent:
        """Basis-Wartungskosten (traditionelle Berechnung)"""
        
        purchase_price = asset_data.get('purchase_price', 100000)
        subcategory = asset_data.get('subcategory', 'Separator')
        drive_type = asset_data.get('drive_type', 'flat - belt drive')
        quality_level = asset_data.get('quality_level', 'standard - Level')
        age_years = asset_data.get('age_years', 1)
        
        # Basis-Wartungsrate nach Zentrifugen-Typ
        if 'separator' in subcategory.lower() or 'clarification' in subcategory.lower():
            base_rate = self.industry_standards['centrifuge_base_maintenance']['disc_stack']
        elif 'decanter' in subcategory.lower():
            base_rate = self.industry_standards['centrifuge_base_maintenance']['decanter']
        else:
            base_rate = self.industry_standards['centrifuge_base_maintenance']['disc_stack']
        
        # Komplexitäts-Faktor
        complexity_factor = self.industry_standards['complexity_factors'][drive_type]['maintenance']
        
        # Qualitäts-Faktor
        quality_factor = self.industry_standards['quality_factors'][quality_level]['maintenance']
        
        # Alters-Faktor (exponentieller Anstieg)
        age_factor = 1.0 + (age_years * 0.08) + (age_years ** 1.3 * 0.015)
        
        # Berechnung
        annual_cost = purchase_price * base_rate * complexity_factor * quality_factor * age_factor
        
        return TCOComponent(
            name='Wartung & Service',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.85,
            calculation_method='price * base_rate * complexity * quality * age',
            factors={
                'purchase_price': purchase_price,
                'base_rate': base_rate,
                'complexity_factor': complexity_factor,
                'quality_factor': quality_factor,
                'age_factor': age_factor
            },
            region_dependent=False,
            equipment_dependent=True
        )
        
    # Neue Methode für ml/tco_components.py ExtendedTCOCalculator Klasse:

def calculate_extended_tco_with_energy_agent(self, asset_data: Dict, lifetime_years: int = 15, energy_agent=None) -> Dict[str, Any]:
    """
    Berechnet erweiterte TCO mit Energy Agent Integration
    
    Args:
        asset_data: Dictionary mit Asset-Eigenschaften
        lifetime_years: Geplante Nutzungsdauer
        energy_agent: EnergyAgent Instanz für Echtzeit-Daten
        
    Returns:
        Dictionary mit detaillierter TCO-Aufschlüsselung
    """
    
    print(f"🔋 Berechne erweiterte TCO mit Energy Agent für {asset_data.get('asset_name', 'Asset')}...")
    
    components = {}
    
    # Standard TCO-Komponenten
    components['maintenance'] = self.add_base_maintenance_component(asset_data)
    components['water'] = self.add_water_component(asset_data)
    components['personnel'] = self.add_personnel_component(asset_data)
    components['spare_parts'] = self.add_spare_parts_component(asset_data)
    components['cleaning'] = self.add_cleaning_component(asset_data)
    components['monitoring'] = self.add_monitoring_component(asset_data)
    components['compliance'] = self.add_compliance_component(asset_data)
    components['insurance'] = self.add_insurance_component(asset_data)
    
    # ENHANCED: Energie-Komponente mit Energy Agent
    if energy_agent:
        components['energy'] = self.add_realtime_energy_component(asset_data, energy_agent)
        print("✅ Echtzeit-Energiepreise integriert")
    else:
        components['energy'] = self.add_energy_component(asset_data)
        print("⚠️ Standard-Energiepreise verwendet")
    
    # Rest der TCO-Berechnung wie gewohnt
    total_annual_operating = sum(comp.annual_cost for comp in components.values())
    escalated_costs = self._calculate_escalated_costs(components, lifetime_years)
    
    # Einmalige Kosten
    purchase_price = asset_data.get('purchase_price', 100000)
    installation_cost = purchase_price * 0.05
    training_cost = purchase_price * 0.02
    disposal_cost = purchase_price * 0.03
    residual_value = purchase_price * 0.15
    
    # TCO-Berechnung
    total_acquisition = purchase_price + installation_cost + training_cost
    total_operating = sum(escalated_costs.values())
    total_disposal = disposal_cost - residual_value
    
    total_tco = total_acquisition + total_operating + total_disposal
    
    # Enhanced Confidence mit Energy Agent
    energy_confidence_bonus = 0.1 if energy_agent else 0.0  # 10% Bonus für Echtzeit-Daten
    total_confidence = sum(comp.confidence * comp.annual_cost for comp in components.values())
    avg_confidence = (total_confidence / total_annual_operating + energy_confidence_bonus) if total_annual_operating > 0 else 0.8
    avg_confidence = min(avg_confidence, 1.0)  # Cap at 100%
    
    # Energy Optimization Insights (falls Energy Agent verfügbar)
    energy_insights = {}
    if energy_agent:
        energy_insights = self.get_energy_optimization_insights(asset_data)
        energy_insights['energy_agent_used'] = True
    else:
        energy_insights['energy_agent_used'] = False
    
    result = {
        'asset_info': {
            'name': asset_data.get('asset_name', 'N/A'),
            'category': f"{asset_data.get('category', 'N/A')} - {asset_data.get('subcategory', 'N/A')}",
            'manufacturer': asset_data.get('manufacturer', 'N/A'),
            'location': asset_data.get('location', 'N/A')
        },
        'components': {name: asdict(comp) for name, comp in components.items()},
        'annual_breakdown': {name: comp.annual_cost for name, comp in components.items()},
        'escalated_costs': escalated_costs,
        'cost_summary': {
            'acquisition_costs': total_acquisition,
            'operating_costs': total_operating,
            'disposal_costs': total_disposal,
            'total_tco': total_tco,
            'annual_average': total_tco / lifetime_years,
            'tco_multiple': total_tco / purchase_price if purchase_price > 0 else 0
        },
        'financial_metrics': {
            'purchase_price': purchase_price,
            'installation_cost': installation_cost,
            'training_cost': training_cost,
            'disposal_cost': disposal_cost,
            'residual_value': residual_value,
            'total_annual_operating': total_annual_operating,
            'lifetime_years': lifetime_years
        },
        'confidence_metrics': {
            'overall_confidence': avg_confidence,
            'confidence_level': self._get_confidence_level(avg_confidence),
            'component_confidence': {name: comp.confidence for name, comp in components.items()},
            'energy_agent_bonus': energy_confidence_bonus
        },
        'energy_insights': energy_insights,  # NEU: Energy-spezifische Insights
        'analysis_metadata': {
            'calculation_date': pd.Timestamp.now().isoformat(),
            'model_version': '2.1_energy_enhanced',
            'regional_factors_applied': asset_data.get('location', 'N/A'),
            'components_count': len(components),
            'energy_agent_used': energy_agent is not None
        }
    }
    
    print(f"✅ Enhanced TCO-Berechnung abgeschlossen: €{total_tco:,.0f} (Konfidenz: {avg_confidence:.1%})")
    if energy_agent:
        print(f"⚡ Energie-Insights: {len(energy_insights.get('recommendations', []))} Optimierungen gefunden")
    
    return result
    
    def add_energy_component(self, asset_data: Dict) -> TCOComponent:
        """Standard-Energiekosten basierend auf Leistungsaufnahme und Betriebszeit"""
        
        total_power_kw = asset_data.get('total_power_consumption', 
                                       asset_data.get('motor_power_kw', 20))
        usage_pattern = asset_data.get('usage_pattern', 'Standard (8h/Tag)')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        efficiency_class = asset_data.get('efficiency_class', 'Standard')
        
        # Betriebsstunden pro Jahr
        annual_hours = self.industry_standards['annual_operating_hours'][usage_pattern]
        
        # Strompreis nach Region
        electricity_price = self.regional_factors['electricity_prices'][location]
        
        # Effizienz-Faktor (Premium-Geräte sind oft effizienter)
        efficiency_factor = 0.95 if efficiency_class == 'Premium' else 1.0
        
        # Load-Faktor (Zentrifugen laufen nicht immer bei Volllast)
        load_factor = {
            'Gelegentlich': 0.6,
            'Standard (8h/Tag)': 0.75,
            'Extended (12h/Tag)': 0.85,
            '24/7 Betrieb': 0.80  # Durchschnittlich wegen variablem Durchsatz
        }.get(usage_pattern, 0.75)
        
        # Berechnung
        annual_kwh = total_power_kw * annual_hours * load_factor * efficiency_factor
        annual_cost = annual_kwh * electricity_price
        
        return TCOComponent(
            name='Energiekosten',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.90,
            calculation_method='power * hours * load_factor * efficiency * price',
            factors={
                'total_power_kw': total_power_kw,
                'annual_hours': annual_hours,
                'load_factor': load_factor,
                'efficiency_factor': efficiency_factor,
                'electricity_price': electricity_price,
                'annual_kwh': annual_kwh
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_realtime_energy_component(self, asset_data: Dict, energy_agent=None) -> TCOComponent:
        """Erweiterte Energiekosten mit Echtzeit-Preisen und Optimierung"""
        
        total_power_kw = asset_data.get('total_power_consumption', 
                                       asset_data.get('motor_power_kw', 20))
        usage_pattern = asset_data.get('usage_pattern', 'Standard (8h/Tag)')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        efficiency_class = asset_data.get('efficiency_class', 'Standard')
        
        # Betriebsstunden pro Jahr
        annual_hours = self.industry_standards['annual_operating_hours'][usage_pattern]
        
        # Echtzeit-Strompreis holen (falls Agent verfügbar)
        if energy_agent:
            try:
                current_price, price_source, is_realtime = energy_agent.get_current_electricity_price(location)
                electricity_price = current_price
                
                # Optimierungsempfehlungen holen
                forecast = energy_agent.get_daily_price_forecast(location, days=1)
                optimization_recommendations = energy_agent.get_optimization_recommendations(asset_data, forecast)
                
                # Store für spätere Verwendung
                asset_data['_energy_optimization'] = optimization_recommendations
                asset_data['_energy_price_realtime'] = is_realtime
                asset_data['_energy_price_source'] = price_source
                
            except Exception as e:
                print(f"⚠️ Energie-Agent Fehler: {e}")
                # Fallback auf regionale Standardpreise
                electricity_price = self.regional_factors['electricity_prices'][location]
                is_realtime = False
                price_source = 'Regional Standard (Agent Error)'
        else:
            # Fallback ohne Agent
            electricity_price = self.regional_factors['electricity_prices'][location]
            is_realtime = False
            price_source = 'Regional Standard'
        
        # Effizienz-Faktor
        efficiency_factor = 0.95 if efficiency_class == 'Premium' else 1.0
        
        # Load-Faktor (realistischer Durchschnittsverbrauch)
        load_factor = {
            'Gelegentlich': 0.6,
            'Standard (8h/Tag)': 0.75,
            'Extended (12h/Tag)': 0.85,
            '24/7 Betrieb': 0.80
        }.get(usage_pattern, 0.75)
        
        # Seasonal Variation (Zentrifugen in Lebensmittel haben Saisons)
        category = asset_data.get('category', 'Industrial')
        seasonal_factor = 1.2 if category in ['Citrus', 'Wine'] else 1.0
        
        # Berechnung
        annual_kwh = total_power_kw * annual_hours * load_factor * efficiency_factor * seasonal_factor
        annual_cost = annual_kwh * electricity_price
        
        # Store annual kWh for later use
        asset_data['_annual_kwh'] = annual_kwh
        asset_data['_last_energy_cost'] = annual_cost
        
        # Confidence höher bei Echtzeit-Preisen
        confidence = 0.95 if is_realtime else 0.85
        
        return TCOComponent(
            name='Energiekosten (Enhanced)',
            annual_cost=annual_cost,
            category='variable',
            confidence=confidence,
            calculation_method='power * hours * load_factor * efficiency * seasonal * realtime_price',
            factors={
                'total_power_kw': total_power_kw,
                'annual_hours': annual_hours,
                'load_factor': load_factor,
                'efficiency_factor': efficiency_factor,
                'seasonal_factor': seasonal_factor,
                'electricity_price': electricity_price,
                'electricity_price_realtime': is_realtime,
                'annual_kwh': annual_kwh,
                'price_source': price_source
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_water_component(self, asset_data: Dict) -> TCOComponent:
        """Wasserkosten für Betrieb und Reinigung"""
        
        water_consumption_ls = asset_data.get('water_consumption_ls', 0.8)
        water_per_ejection = asset_data.get('water_per_ejection', 2.0)
        usage_pattern = asset_data.get('usage_pattern', 'Standard (8h/Tag)')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        category = asset_data.get('category', 'Industrial')
        
        # Betriebsstunden pro Jahr
        annual_hours = self.industry_standards['annual_operating_hours'][usage_pattern]
        
        # Wasserpreis nach Region
        water_price = self.regional_factors['water_prices'][location]
        
        # Ejektions-Häufigkeit (abhängig von Anwendung)
        ejections_per_hour = {
            'Citrus': 4,     # Häufige Reinigung
            'Wine': 2,       # Moderate Reinigung
            'Dairy': 6,      # Sehr häufige Reinigung
            'Industrial': 3   # Standard
        }.get(category, 3)
        
        # CIP-Reinigung (zusätzlich bei Lebensmitteln)
        cip_water_factor = 1.5 if category in ['Citrus', 'Wine', 'Dairy'] else 1.0
        
        # Berechnung
        # Betriebswasser + Ejektionswasser + CIP-Reinigung
        hourly_water = (water_consumption_ls + 
                       (water_per_ejection * ejections_per_hour)) * cip_water_factor
        
        annual_water_liters = hourly_water * annual_hours
        annual_cost = annual_water_liters * water_price
        
        return TCOComponent(
            name='Wasserkosten',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.80,
            calculation_method='(operation + ejection) * cip_factor * hours * price',
            factors={
                'water_consumption_ls': water_consumption_ls,
                'water_per_ejection': water_per_ejection,
                'ejections_per_hour': ejections_per_hour,
                'cip_water_factor': cip_water_factor,
                'annual_hours': annual_hours,
                'water_price': water_price,
                'annual_water_liters': annual_water_liters
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_personnel_component(self, asset_data: Dict) -> TCOComponent:
        """Personalkosten für Bedienung und Wartung"""
        
        drive_type = asset_data.get('drive_type', 'flat - belt drive')
        quality_level = asset_data.get('quality_level', 'standard - Level')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        criticality = asset_data.get('criticality', 'Mittel')
        category = asset_data.get('category', 'Industrial')
        
        # Basis-Bedienerstunden pro Jahr
        base_hours = self.industry_standards['quality_factors'][quality_level]['personnel'] * 400
        
        # Komplexitäts-Faktor
        complexity_factor = self.industry_standards['complexity_factors'][drive_type]['personnel']
        
        # Kritikalitäts-Faktor (kritische Assets brauchen mehr Aufmerksamkeit)
        criticality_factor = {
            'Niedrig': 0.8,
            'Mittel': 1.0,
            'Hoch': 1.3,
            'Kritisch': 1.6
        }.get(criticality, 1.0)
        
        # Lebensmittel-Faktor (mehr Hygiene-Aufwand)
        food_factor = 1.4 if category in ['Citrus', 'Wine', 'Dairy'] else 1.0
        
        # Stundenlohn nach Region
        hourly_wage = self.regional_factors['labor_costs'][location]
        
        # Berechnung
        total_hours = base_hours * complexity_factor * criticality_factor * food_factor
        annual_cost = total_hours * hourly_wage
        
        return TCOComponent(
            name='Personalkosten',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.75,
            calculation_method='base_hours * complexity * criticality * food_factor * wage',
            factors={
                'base_hours': base_hours,
                'complexity_factor': complexity_factor,
                'criticality_factor': criticality_factor,
                'food_factor': food_factor,
                'hourly_wage': hourly_wage,
                'total_hours': total_hours
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_spare_parts_component(self, asset_data: Dict) -> TCOComponent:
        """Ersatzteilkosten basierend auf Verschleiß und Verfügbarkeit"""
        
        purchase_price = asset_data.get('purchase_price', 100000)
        quality_level = asset_data.get('quality_level', 'standard - Level')
        usage_pattern = asset_data.get('usage_pattern', 'Standard (8h/Tag)')
        age_years = asset_data.get('age_years', 1)
        manufacturer = asset_data.get('manufacturer', 'GEA')
        
        # Basis-Ersatzteilkostenrate
        base_spare_parts_rate = 0.04  # 4% vom Anschaffungspreis
        
        # Premium-Geräte haben oft teurere, aber länger haltende Teile
        quality_factor = 1.3 if quality_level == 'premium - Level' else 1.0
        
        # Nutzungsintensitäts-Faktor
        usage_factor = {
            'Gelegentlich': 0.6,
            'Standard (8h/Tag)': 1.0,
            'Extended (12h/Tag)': 1.4,
            '24/7 Betrieb': 2.0
        }.get(usage_pattern, 1.0)
        
        # Alters-Faktor (mehr Verschleiß mit der Zeit)
        age_factor = 1.0 + (age_years * 0.12)
        
        # Hersteller-Faktor (Markenhersteller = teurere Teile)
        manufacturer_factor = 1.2 if manufacturer in ['GEA', 'Alfa Laval'] else 1.0
        
        # Berechnung
        annual_cost = (purchase_price * base_spare_parts_rate * quality_factor * 
                      usage_factor * age_factor * manufacturer_factor)
        
        return TCOComponent(
            name='Ersatzteile',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.70,
            calculation_method='price * base_rate * quality * usage * age * manufacturer',
            factors={
                'purchase_price': purchase_price,
                'base_spare_parts_rate': base_spare_parts_rate,
                'quality_factor': quality_factor,
                'usage_factor': usage_factor,
                'age_factor': age_factor,
                'manufacturer_factor': manufacturer_factor
            },
            region_dependent=False,
            equipment_dependent=True
        )
    
    def add_cleaning_component(self, asset_data: Dict) -> TCOComponent:
        """Reinigungs- und Hygienekosten (besonders für Lebensmittel)"""
        
        purchase_price = asset_data.get('purchase_price', 100000)
        category = asset_data.get('category', 'Industrial')
        usage_pattern = asset_data.get('usage_pattern', 'Standard (8h/Tag)')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        
        # Nur relevant für Lebensmittelanwendungen
        if category not in ['Citrus', 'Wine', 'Dairy']:
            return TCOComponent(
                name='Reinigung & Hygiene',
                annual_cost=0,
                category='variable',
                confidence=1.0,
                calculation_method='not_applicable',
                factors={},
                region_dependent=False,
                equipment_dependent=False
            )
        
        # Basis-Reinigungskosten
        base_cleaning_rate = {
            'Citrus': 0.025,   # 2.5% - mittlere Hygieneanforderungen
            'Wine': 0.02,      # 2.0% - moderate Hygieneanforderungen  
            'Dairy': 0.035     # 3.5% - höchste Hygieneanforderungen
        }.get(category, 0.02)
        
        # Nutzungsintensität beeinflusst Reinigungsfrequenz
        usage_factor = {
            'Gelegentlich': 0.7,
            'Standard (8h/Tag)': 1.0,
            'Extended (12h/Tag)': 1.3,
            '24/7 Betrieb': 1.6
        }.get(usage_pattern, 1.0)
        
        # Regionale Faktoren für Chemikalien und Arbeit
        regional_factor = self.regional_factors['regulatory_compliance'][location]
        
        # CIP-Chemikalien, Arbeitszeit, Validierung
        annual_cost = purchase_price * base_cleaning_rate * usage_factor * regional_factor
        
        return TCOComponent(
            name='Reinigung & Hygiene',
            annual_cost=annual_cost,
            category='variable',
            confidence=0.80,
            calculation_method='price * cleaning_rate * usage * regional',
            factors={
                'purchase_price': purchase_price,
                'base_cleaning_rate': base_cleaning_rate,
                'usage_factor': usage_factor,
                'regional_factor': regional_factor
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_monitoring_component(self, asset_data: Dict) -> TCOComponent:
        """IoT-Monitoring und Predictive Maintenance"""
        
        criticality = asset_data.get('criticality', 'Mittel')
        purchase_price = asset_data.get('purchase_price', 100000)
        
        # Basis-Monitoring-Kosten nach Kritikalität
        base_monitoring_cost = self.industry_standards['criticality_factors'][criticality]['monitoring']
        
        # Zusätzliche Cloud/Software-Kosten für größere Anlagen
        if purchase_price > 200000:
            base_monitoring_cost += 1500  # Premium Cloud-Services
        
        # Jährliche Software-Lizenzen
        software_cost = base_monitoring_cost * 0.3 if base_monitoring_cost > 0 else 0
        
        annual_cost = base_monitoring_cost + software_cost
        
        return TCOComponent(
            name='Monitoring & IoT',
            annual_cost=annual_cost,
            category='fixed',
            confidence=0.85,
            calculation_method='base_cost + software_licenses',
            factors={
                'criticality': criticality,
                'base_monitoring_cost': base_monitoring_cost,
                'software_cost': software_cost
            },
            region_dependent=False,
            equipment_dependent=True
        )
    
    def add_compliance_component(self, asset_data: Dict) -> TCOComponent:
        """Compliance und Zertifizierungskosten"""
        
        category = asset_data.get('category', 'Industrial')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        purchase_price = asset_data.get('purchase_price', 100000)
        
        # Basis-Compliance-Kosten
        base_compliance_cost = 0
        
        # Lebensmittel haben höhere Compliance-Anforderungen
        if category in ['Citrus', 'Wine', 'Dairy']:
            base_compliance_cost = 2500  # HACCP, FDA, EU-Verordnungen
        else:
            base_compliance_cost = 1000  # Grundlegende Sicherheitsstandards
        
        # Regionale Compliance-Faktoren
        regional_factor = self.regional_factors['regulatory_compliance'][location]
        
        # Größenfaktor (größere Anlagen = mehr Aufwand)
        size_factor = 1.0 + (purchase_price / 500000) * 0.5  # bis 50% Aufschlag
        
        annual_cost = base_compliance_cost * regional_factor * size_factor
        
        return TCOComponent(
            name='Compliance & Zertifizierung',
            annual_cost=annual_cost,
            category='fixed',
            confidence=0.75,
            calculation_method='base_cost * regional * size_factor',
            factors={
                'base_compliance_cost': base_compliance_cost,
                'regional_factor': regional_factor,
                'size_factor': size_factor
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def add_insurance_component(self, asset_data: Dict) -> TCOComponent:
        """Versicherungskosten basierend auf Anlagenwert und Risiko"""
        
        purchase_price = asset_data.get('purchase_price', 100000)
        criticality = asset_data.get('criticality', 'Mittel')
        location = asset_data.get('location', 'Düsseldorf (HQ)')
        category = asset_data.get('category', 'Industrial')
        
        # Basis-Versicherungsrate
        base_insurance_rate = 0.008  # 0.8% vom Anlagewert
        
        # Kritikalitäts-Faktor
        criticality_factor = {
            'Niedrig': 0.8,
            'Mittel': 1.0,
            'Hoch': 1.3,
            'Kritisch': 1.6
        }.get(criticality, 1.0)
        
        # Lebensmittel = höheres Haftungsrisiko
        category_factor = 1.2 if category in ['Citrus', 'Wine', 'Dairy'] else 1.0
        
        # Regionale Versicherungskosten
        regional_factor = {
            'Deutschland': 1.0, 'Dänemark': 0.95, 'Italien': 1.1,
            'China': 1.3, 'USA': 1.2, 'Brasilien': 1.4
        }.get(location.split(' ')[0] if ' ' in location else location, 1.0)
        
        annual_cost = purchase_price * base_insurance_rate * criticality_factor * category_factor * regional_factor
        
        return TCOComponent(
            name='Versicherung',
            annual_cost=annual_cost,
            category='fixed',
            confidence=0.90,
            calculation_method='price * rate * criticality * category * regional',
            factors={
                'purchase_price': purchase_price,
                'base_insurance_rate': base_insurance_rate,
                'criticality_factor': criticality_factor,
                'category_factor': category_factor,
                'regional_factor': regional_factor
            },
            region_dependent=True,
            equipment_dependent=True
        )
    
    def calculate_extended_tco(self, asset_data: Dict, lifetime_years: int = 15) -> Dict[str, Any]:
        """
        Berechnet komplette erweiterte TCO mit allen Komponenten
        
        Args:
            asset_data: Dictionary mit Asset-Eigenschaften
            lifetime_years: Geplante Nutzungsdauer
            
        Returns:
            Dictionary mit detaillierter TCO-Aufschlüsselung
        """
        
        print(f"🧮 Berechne erweiterte TCO für {asset_data.get('asset_name', 'Asset')}...")
        
        components = {}
        
        # Alle TCO-Komponenten berechnen
        components['maintenance'] = self.add_base_maintenance_component(asset_data)
        components['energy'] = self.add_energy_component(asset_data)
        components['water'] = self.add_water_component(asset_data)
        components['personnel'] = self.add_personnel_component(asset_data)
        components['spare_parts'] = self.add_spare_parts_component(asset_data)
        components['cleaning'] = self.add_cleaning_component(asset_data)
        components['monitoring'] = self.add_monitoring_component(asset_data)
        components['compliance'] = self.add_compliance_component(asset_data)
        components['insurance'] = self.add_insurance_component(asset_data)
        
        # Gesamte jährliche Betriebskosten
        total_annual_operating = sum(comp.annual_cost for comp in components.values())
        
        # Escalation über Lebensdauer (Inflation, Verschleiß)
        escalated_costs = self._calculate_escalated_costs(components, lifetime_years)
        
        # Einmalige Kosten
        purchase_price = asset_data.get('purchase_price', 100000)
        installation_cost = purchase_price * 0.05  # 5% für Installation
        training_cost = purchase_price * 0.02      # 2% für Training
        disposal_cost = purchase_price * 0.03      # 3% für Entsorgung (Endwert)
        
        # Restwert am Ende der Nutzung
        residual_value = purchase_price * 0.15  # 15% Restwert nach 15 Jahren
        
        # TCO-Berechnung
        total_acquisition = purchase_price + installation_cost + training_cost
        total_operating = sum(escalated_costs.values())
        total_disposal = disposal_cost - residual_value
        
        total_tco = total_acquisition + total_operating + total_disposal
        
        # Confidence Score (gewichteter Durchschnitt)
        total_confidence = sum(comp.confidence * comp.annual_cost for comp in components.values())
        avg_confidence = total_confidence / total_annual_operating if total_annual_operating > 0 else 0.8
        
        result = {
            'asset_info': {
                'name': asset_data.get('asset_name', 'N/A'),
                'category': f"{asset_data.get('category', 'N/A')} - {asset_data.get('subcategory', 'N/A')}",
                'manufacturer': asset_data.get('manufacturer', 'N/A'),
                'location': asset_data.get('location', 'N/A')
            },
            'components': {name: asdict(comp) for name, comp in components.items()},
            'annual_breakdown': {name: comp.annual_cost for name, comp in components.items()},
            'escalated_costs': escalated_costs,
            'cost_summary': {
                'acquisition_costs': total_acquisition,
                'operating_costs': total_operating,
                'disposal_costs': total_disposal,
                'total_tco': total_tco,
                'annual_average': total_tco / lifetime_years,
                'tco_multiple': total_tco / purchase_price if purchase_price > 0 else 0
            },
            'financial_metrics': {
                'purchase_price': purchase_price,
                'installation_cost': installation_cost,
                'training_cost': training_cost,
                'disposal_cost': disposal_cost,
                'residual_value': residual_value,
                'total_annual_operating': total_annual_operating,
                'lifetime_years': lifetime_years
            },
            'confidence_metrics': {
                'overall_confidence': avg_confidence,
                'confidence_level': self._get_confidence_level(avg_confidence),
                'component_confidence': {name: comp.confidence for name, comp in components.items()}
            },
            'analysis_metadata': {
                'calculation_date': pd.Timestamp.now().isoformat(),
                'model_version': '2.0_extended',
                'regional_factors_applied': asset_data.get('location', 'N/A'),
                'components_count': len(components)
            }
        }
        
        # Speichere Berechnung für Audit Trail
        self.calculation_history.append({
            'timestamp': pd.Timestamp.now(),
            'asset_name': asset_data.get('asset_name', 'Unknown'),
            'total_tco': total_tco,
            'confidence': avg_confidence
        })
        
        print(f"✅ TCO-Berechnung abgeschlossen: €{total_tco:,.0f} (Konfidenz: {avg_confidence:.1%})")
        
        return result
    
    def calculate_extended_tco_with_energy_agent(self, asset_data: Dict, lifetime_years: int = 15, energy_agent=None) -> Dict[str, Any]:
        """
        Berechnet erweiterte TCO mit Energy Agent Integration
        
        Args:
            asset_data: Dictionary mit Asset-Eigenschaften
            lifetime_years: Geplante Nutzungsdauer
            energy_agent: EnergyAgent Instanz für Echtzeit-Daten
            
        Returns:
            Dictionary mit detaillierter TCO-Aufschlüsselung
        """
        
        print(f"🔋 Berechne erweiterte TCO mit Energy Agent für {asset_data.get('asset_name', 'Asset')}...")
        
        components = {}
        
        # Standard TCO-Komponenten
        components['maintenance'] = self.add_base_maintenance_component(asset_data)
        components['water'] = self.add_water_component(asset_data)
        components['personnel'] = self.add_personnel_component(asset_data)
        components['spare_parts'] = self.add_spare_parts_component(asset_data)
        components['cleaning'] = self.add_cleaning_component(asset_data)
        components['monitoring'] = self.add_monitoring_component(asset_data)
        components['compliance'] = self.add_compliance_component(asset_data)
        components['insurance'] = self.add_insurance_component(asset_data)
        
        # ENHANCED: Energie-Komponente mit Energy Agent
        if energy_agent:
            components['energy'] = self.add_realtime_energy_component(asset_data, energy_agent)
            print("✅ Echtzeit-Energiepreise integriert")
        else:
            components['energy'] = self.add_energy_component(asset_data)
            print("⚠️ Standard-Energiepreise verwendet")
        
        # Rest der TCO-Berechnung wie gewohnt
        total_annual_operating = sum(comp.annual_cost for comp in components.values())
        escalated_costs = self._calculate_escalated_costs(components, lifetime_years)
        
        # Einmalige Kosten
        purchase_price = asset_data.get('purchase_price', 100000)
        installation_cost = purchase_price * 0.05
        training_cost = purchase_price * 0.02
        disposal_cost = purchase_price * 0.03
        residual_value = purchase_price * 0.15
        
        # TCO-Berechnung
        total_acquisition = purchase_price + installation_cost + training_cost
        total_operating = sum(escalated_costs.values())
        total_disposal = disposal_cost - residual_value
        
        total_tco = total_acquisition + total_operating + total_disposal
        
        # Enhanced Confidence mit Energy Agent
        energy_confidence_bonus = 0.1 if energy_agent else 0.0  # 10% Bonus für Echtzeit-Daten
        total_confidence = sum(comp.confidence * comp.annual_cost for comp in components.values())
        avg_confidence = (total_confidence / total_annual_operating + energy_confidence_bonus) if total_annual_operating > 0 else 0.8
        avg_confidence = min(avg_confidence, 1.0)  # Cap at 100%
        
        # Energy Optimization Insights (falls Energy Agent verfügbar)
        energy_insights = {}
        if energy_agent:
            energy_insights = self.get_energy_optimization_insights(asset_data)
            energy_insights['energy_agent_used'] = True
        else:
            energy_insights['energy_agent_used'] = False
        
        result = {
            'asset_info': {
                'name': asset_data.get('asset_name', 'N/A'),
                'category': f"{asset_data.get('category', 'N/A')} - {asset_data.get('subcategory', 'N/A')}",
                'manufacturer': asset_data.get('manufacturer', 'N/A'),
                'location': asset_data.get('location', 'N/A')
            },
            'components': {name: asdict(comp) for name, comp in components.items()},
            'annual_breakdown': {name: comp.annual_cost for name, comp in components.items()},
            'escalated_costs': escalated_costs,
            'cost_summary': {
                'acquisition_costs': total_acquisition,
                'operating_costs': total_operating,
                'disposal_costs': total_disposal,
                'total_tco': total_tco,
                'annual_average': total_tco / lifetime_years,
                'tco_multiple': total_tco / purchase_price if purchase_price > 0 else 0
            },
            'financial_metrics': {
                'purchase_price': purchase_price,
                'installation_cost': installation_cost,
                'training_cost': training_cost,
                'disposal_cost': disposal_cost,
                'residual_value': residual_value,
                'total_annual_operating': total_annual_operating,
                'lifetime_years': lifetime_years
            },
            'confidence_metrics': {
                'overall_confidence': avg_confidence,
                'confidence_level': self._get_confidence_level(avg_confidence),
                'component_confidence': {name: comp.confidence for name, comp in components.items()},
                'energy_agent_bonus': energy_confidence_bonus
            },
            'energy_insights': energy_insights,  # NEU: Energy-spezifische Insights
            'analysis_metadata': {
                'calculation_date': pd.Timestamp.now().isoformat(),
                'model_version': '2.1_energy_enhanced',
                'regional_factors_applied': asset_data.get('location', 'N/A'),
                'components_count': len(components),
                'energy_agent_used': energy_agent is not None
            }
        }
        
        print(f"✅ Enhanced TCO-Berechnung abgeschlossen: €{total_tco:,.0f} (Konfidenz: {avg_confidence:.1%})")
        if energy_agent:
            print(f"⚡ Energie-Insights: {len(energy_insights.get('recommendations', []))} Optimierungen gefunden")
        
        return result
    
    def get_energy_optimization_insights(self, asset_data: Dict) -> Dict[str, Any]:
        """Gibt detaillierte Energie-Optimierungs-Insights zurück"""
        
        optimization_recommendations = asset_data.get('_energy_optimization', [])
        energy_cost = asset_data.get('_last_energy_cost', 0)
        
        insights = {
            'current_energy_cost': energy_cost,
            'optimization_count': len(optimization_recommendations),
            'total_savings_potential': sum(rec.get('potential_savings', 0) for rec in optimization_recommendations if isinstance(rec.get('potential_savings'), (int, float))),
            'recommendations': optimization_recommendations,
            'energy_efficiency_rating': 'Standard',  # Default
            'smart_grid_ready': False
        }
        
        # Energy Efficiency Rating basierend auf kWh/€ Purchase Price
        annual_kwh = asset_data.get('_annual_kwh', 0)
        purchase_price = asset_data.get('purchase_price', 1)
        
        if annual_kwh > 0 and purchase_price > 0:
            kwh_per_euro = annual_kwh / purchase_price
            
            if kwh_per_euro < 0.5:
                insights['energy_efficiency_rating'] = 'Excellent'
            elif kwh_per_euro < 1.0:
                insights['energy_efficiency_rating'] = 'Good'
            elif kwh_per_euro < 2.0:
                insights['energy_efficiency_rating'] = 'Average'
            else:
                insights['energy_efficiency_rating'] = 'Poor'
        
        # Smart Grid Readiness
        power_kw = asset_data.get('total_power_consumption', 0)
        if power_kw > 50:  # Größere Anlagen für Demand Response geeignet
            insights['smart_grid_ready'] = True
        
        return insights
    
    def _calculate_escalated_costs(self, components: Dict[str, TCOComponent], lifetime_years: int) -> Dict[str, float]:
        """Berechnet eskalierte Kosten über Lebensdauer"""
        
        escalated = {}
        
        for name, component in components.items():
            if component.category == 'variable':
                # Variable Kosten steigen mit Inflation und Verschleiß
                annual_escalation = 0.03  # 3% Inflation
                wear_escalation = 0.02    # 2% zusätzlicher Verschleiß
                
                total_cost = 0
                for year in range(1, lifetime_years + 1):
                    year_factor = (1 + annual_escalation + wear_escalation) ** (year - 1)
                    total_cost += component.annual_cost * year_factor
                
                escalated[name] = total_cost
                
            elif component.category == 'fixed':
                # Fixe Kosten nur mit Inflation
                annual_escalation = 0.03
                
                total_cost = 0
                for year in range(1, lifetime_years + 1):
                    year_factor = (1 + annual_escalation) ** (year - 1)
                    total_cost += component.annual_cost * year_factor
                
                escalated[name] = total_cost
                
            else:  # one_time
                escalated[name] = component.annual_cost
        
        return escalated
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Konvertiert numerische Konfidenz in Level"""
        if confidence >= 0.85:
            return "Sehr Hoch"
        elif confidence >= 0.75:
            return "Hoch"
        elif confidence >= 0.65:
            return "Mittel"
        else:
            return "Niedrig"
    
    def generate_tco_report(self, tco_result: Dict) -> str:
        """Generiert einen formatierten TCO-Report"""
        
        asset_info = tco_result['asset_info']
        cost_summary = tco_result['cost_summary']
        annual_breakdown = tco_result['annual_breakdown']
        confidence = tco_result['confidence_metrics']
        
        report = f"""
# TCO-ANALYSE REPORT
## {asset_info['name']}

**Asset Details:**
- Kategorie: {asset_info['category']}
- Hersteller: {asset_info['manufacturer']}
- Standort: {asset_info['location']}

**Kosten-Zusammenfassung:**
- Anschaffungskosten: €{cost_summary['acquisition_costs']:,.0f}
- Betriebskosten (gesamt): €{cost_summary['operating_costs']:,.0f}
- Entsorgungskosten: €{cost_summary['disposal_costs']:,.0f}
- **GESAMT-TCO: €{cost_summary['total_tco']:,.0f}**
- Durchschnitt/Jahr: €{cost_summary['annual_average']:,.0f}
- TCO-Multiplikator: {cost_summary['tco_multiple']:.1f}x

**Jährliche Betriebskosten-Aufschlüsselung:**
"""
        
        for component, cost in sorted(annual_breakdown.items(), key=lambda x: x[1], reverse=True):
            if cost > 0:
                percentage = (cost / sum(annual_breakdown.values())) * 100
                report += f"- {component.title()}: €{cost:,.0f} ({percentage:.1f}%)\n"
        
        report += f"""
**Analyse-Qualität:**
- Gesamtkonfidenz: {confidence['overall_confidence']:.1%} ({confidence['confidence_level']})
- Berechnet am: {tco_result['analysis_metadata']['calculation_date'][:10]}
"""
        
        return report
    
    def export_to_excel(self, tco_result: Dict, filepath: str = None):
        """Exportiert TCO-Analyse nach Excel"""
        
        import pandas as pd
        from datetime import datetime
        
        if filepath is None:
            asset_name = tco_result['asset_info']['name'].replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filepath = f"TCO_Analysis_{asset_name}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # Sheet 1: Summary
            summary_data = {
                'Kategorie': ['Anschaffung', 'Betrieb (Gesamt)', 'Entsorgung', 'GESAMT-TCO'],
                'Kosten (€)': [
                    tco_result['cost_summary']['acquisition_costs'],
                    tco_result['cost_summary']['operating_costs'],
                    tco_result['cost_summary']['disposal_costs'],
                    tco_result['cost_summary']['total_tco']
                ],
                'Anteil (%)': [
                    tco_result['cost_summary']['acquisition_costs'] / tco_result['cost_summary']['total_tco'] * 100,
                    tco_result['cost_summary']['operating_costs'] / tco_result['cost_summary']['total_tco'] * 100,
                    tco_result['cost_summary']['disposal_costs'] / tco_result['cost_summary']['total_tco'] * 100,
                    100.0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='TCO_Summary', index=False)
            
            # Sheet 2: Annual Breakdown
            breakdown_data = []
            for component, cost in tco_result['annual_breakdown'].items():
                if cost > 0:
                    breakdown_data.append({
                        'Komponente': component.replace('_', ' ').title(),
                        'Jährliche_Kosten_€': cost,
                        'Anteil_%': (cost / sum(tco_result['annual_breakdown'].values())) * 100,
                        'Kategorie': tco_result['components'][component]['category'],
                        'Konfidenz_%': tco_result['components'][component]['confidence'] * 100,
                        'Regional_abhängig': tco_result['components'][component]['region_dependent']
                    })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            breakdown_df.to_excel(writer, sheet_name='Annual_Breakdown', index=False)
            
            # Sheet 3: Lifetime Escalation
            escalated_data = []
            for component, total_cost in tco_result['escalated_costs'].items():
                if total_cost > 0:
                    escalated_data.append({
                        'Komponente': component.replace('_', ' ').title(),
                        'Gesamtkosten_Lebensdauer_€': total_cost,
                        'Jährliche_Basis_€': tco_result['annual_breakdown'][component],
                        'Escalation_Faktor': total_cost / (tco_result['annual_breakdown'][component] * tco_result['financial_metrics']['lifetime_years']) if tco_result['annual_breakdown'][component] > 0 else 1.0
                    })
            
            escalated_df = pd.DataFrame(escalated_data)
            escalated_df.to_excel(writer, sheet_name='Lifetime_Costs', index=False)
            
            # Sheet 4: Asset Info & Metadata
            metadata = {
                'Parameter': ['Asset Name', 'Kategorie', 'Hersteller', 'Standort', 'Anschaffungspreis', 
                             'Nutzungsdauer', 'Berechnet am', 'Model Version', 'Konfidenz'],
                'Wert': [
                    tco_result['asset_info']['name'],
                    tco_result['asset_info']['category'],
                    tco_result['asset_info']['manufacturer'],
                    tco_result['asset_info']['location'],
                    f"€{tco_result['financial_metrics']['purchase_price']:,.0f}",
                    f"{tco_result['financial_metrics']['lifetime_years']} Jahre",
                    tco_result['analysis_metadata']['calculation_date'][:10],
                    tco_result['analysis_metadata']['model_version'],
                    f"{tco_result['confidence_metrics']['overall_confidence']:.1%}"
                ]
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Asset_Info', index=False)
        
        print(f"📊 TCO-Analyse exportiert nach: {filepath}")
        return filepath
    
    def compare_assets(self, asset_list: List[Dict]) -> pd.DataFrame:
        """Vergleicht mehrere Assets und gibt Vergleichstabelle zurück"""
        
        comparison_data = []
        
        for asset_data in asset_list:
            # TCO für jedes Asset berechnen
            tco_result = self.calculate_extended_tco(asset_data)
            
            comparison_data.append({
                'Asset_Name': asset_data.get('asset_name', 'N/A'),
                'Kategorie': asset_data.get('category', 'N/A'),
                'Hersteller': asset_data.get('manufacturer', 'N/A'),
                'Anschaffungspreis_€': asset_data.get('purchase_price', 0),
                'Jährliche_Betriebskosten_€': tco_result['financial_metrics']['total_annual_operating'],
                'Gesamt_TCO_€': tco_result['cost_summary']['total_tco'],
                'TCO_Multiplikator': tco_result['cost_summary']['tco_multiple'],
                'Energie_€_Jahr': tco_result['annual_breakdown'].get('energy', 0),
                'Wasser_€_Jahr': tco_result['annual_breakdown'].get('water', 0),
                'Personal_€_Jahr': tco_result['annual_breakdown'].get('personnel', 0),
                'Wartung_€_Jahr': tco_result['annual_breakdown'].get('maintenance', 0),
                'Konfidenz_%': tco_result['confidence_metrics']['overall_confidence'] * 100,
                'Standort': asset_data.get('location', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ranking hinzufügen
        comparison_df['TCO_Ranking'] = comparison_df['Gesamt_TCO_€'].rank()
        comparison_df['Effizienz_Ranking'] = (comparison_df['Jährliche_Betriebskosten_€'] / comparison_df['Anschaffungspreis_€']).rank()
        
        return comparison_df.sort_values('TCO_Ranking')
    
    def get_optimization_recommendations(self, tco_result: Dict) -> List[Dict]:
        """Generiert Optimierungs-Empfehlungen basierend auf TCO-Analyse"""
        
        recommendations = []
        annual_breakdown = tco_result['annual_breakdown']
        total_annual = sum(annual_breakdown.values())
        asset_info = tco_result['asset_info']
        
        # Energie-Optimierung
        energy_cost = annual_breakdown.get('energy', 0)
        if energy_cost > total_annual * 0.15:  # >15% der Betriebskosten
            energy_saving_potential = energy_cost * 0.25  # 25% durch High-Efficiency
            recommendations.append({
                'priority': 'Hoch',
                'category': 'Energie-Effizienz',
                'title': 'High-Efficiency Motor Upgrade',
                'description': f'Energiekosten machen {(energy_cost/total_annual)*100:.0f}% der Betriebskosten aus.',
                'current_cost': energy_cost,
                'potential_savings': energy_saving_potential,
                'payback_period': 'Investition von €15-25k amortisiert sich in 2-3 Jahren',
                'implementation': 'IE4 Premium-Motor oder Frequenzumrichter installieren',
                'confidence': 0.85
            })
        
        # Personal-Optimierung
        personnel_cost = annual_breakdown.get('personnel', 0)
        if personnel_cost > 15000:  # >€15k/Jahr
            personnel_saving = personnel_cost * 0.30  # 30% durch Automatisierung
            recommendations.append({
                'priority': 'Mittel',
                'category': 'Automatisierung',
                'title': 'IoT-Monitoring & Auto-CIP',
                'description': f'Personalkosten von €{personnel_cost:,.0f}/Jahr durch Automatisierung reduzieren.',
                'current_cost': personnel_cost,
                'potential_savings': personnel_saving,
                'payback_period': 'IoT-Investment von €25-50k amortisiert sich in 3-4 Jahren',
                'implementation': 'Condition Monitoring und automatische CIP-Zyklen',
                'confidence': 0.70
            })
        
        # Wartungs-Optimierung
        maintenance_cost = annual_breakdown.get('maintenance', 0)
        spare_parts_cost = annual_breakdown.get('spare_parts', 0)
        if (maintenance_cost + spare_parts_cost) > total_annual * 0.25:  # >25%
            maintenance_saving = (maintenance_cost + spare_parts_cost) * 0.20  # 20% durch Predictive
            recommendations.append({
                'priority': 'Hoch',
                'category': 'Predictive Maintenance',
                'title': 'Condition-Based Maintenance',
                'description': f'Wartungskosten von €{maintenance_cost + spare_parts_cost:,.0f} durch Predictive Maintenance optimieren.',
                'current_cost': maintenance_cost + spare_parts_cost,
                'potential_savings': maintenance_saving,
                'payback_period': 'Sensor-Investment von €15-30k amortisiert sich in 2-3 Jahren',
                'implementation': 'Vibrations-, Temperatur- und Ölanalyse-Sensoren',
                'confidence': 0.80
            })
        
        # Wasser-Optimierung (für Lebensmittel)
        water_cost = annual_breakdown.get('water', 0)
        cleaning_cost = annual_breakdown.get('cleaning', 0)
        if (water_cost + cleaning_cost) > 5000:  # >€5k/Jahr
            water_saving = (water_cost + cleaning_cost) * 0.15  # 15% durch Optimierung
            recommendations.append({
                'priority': 'Niedrig',
                'category': 'Wassereffizienz',
                'title': 'CIP-Optimierung & Wasserrecycling',
                'description': f'Wasser- und Reinigungskosten von €{water_cost + cleaning_cost:,.0f} reduzieren.',
                'current_cost': water_cost + cleaning_cost,
                'potential_savings': water_saving,
                'payback_period': 'Water-Recovery System amortisiert sich in 4-6 Jahren',
                'implementation': 'Optimierte CIP-Zyklen und Wasserrecycling-System',
                'confidence': 0.65
            })
        
        # Standort-Optimierung (bei hohen regionalen Kosten)
        location = asset_info.get('location', '')
        if 'Düsseldorf' in location or 'Kopenhagen' in location:  # Hochkosten-Standorte
            personnel_cost = annual_breakdown.get('personnel', 0)
            recommendations.append({
                'priority': 'Strategisch',
                'category': 'Standort-Strategie',
                'title': 'Remote Monitoring implementieren',
                'description': f'Hohe Personalkosten am Standort {location} durch Remote Services reduzieren.',
                'current_cost': personnel_cost,
                'potential_savings': personnel_cost * 0.20,  # 20% durch Remote
                'payback_period': 'Remote-Infrastruktur amortisiert sich in 2-3 Jahren',
                'implementation': 'Zentrale Service-Leitstelle mit Remote-Diagnose',
                'confidence': 0.60
            })
        
        # Sortiere nach Priorität und Einsparungspotential
        priority_order = {'Hoch': 3, 'Mittel': 2, 'Niedrig': 1, 'Strategisch': 0}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['potential_savings']), reverse=True)
        
        return recommendations[:5]  # Top 5 Empfehlungen
    
    def get_benchmark_comparison(self, asset_data: Dict, tco_result: Dict) -> Dict:
        """Vergleicht Asset mit Industrie-Benchmarks"""
        
        category = asset_data.get('category', 'Industrial')
        subcategory = asset_data.get('subcategory', 'Separator')
        purchase_price = asset_data.get('purchase_price', 100000)
        
        # Industrie-Benchmarks (basierend auf GEA-Erfahrungen)
        benchmarks = {
            'Citrus': {
                'maintenance_ratio': 0.12,      # 12% vom Anschaffungspreis
                'energy_ratio': 0.15,           # 15% der Betriebskosten
                'water_ratio': 0.08,            # 8% der Betriebskosten
                'tco_multiple': 4.5,            # 4.5x Anschaffungspreis über 15 Jahre
                'availability': 0.95             # 95% Verfügbarkeit
            },
            'Wine': {
                'maintenance_ratio': 0.10,
                'energy_ratio': 0.12,
                'water_ratio': 0.06,
                'tco_multiple': 4.0,
                'availability': 0.93
            },
            'Dairy': {
                'maintenance_ratio': 0.14,
                'energy_ratio': 0.18,
                'water_ratio': 0.12,
                'tco_multiple': 5.2,
                'availability': 0.97
            },
            'Industrial': {
                'maintenance_ratio': 0.11,
                'energy_ratio': 0.14,
                'water_ratio': 0.05,
                'tco_multiple': 4.2,
                'availability': 0.94
            }
        }
        
        benchmark = benchmarks.get(category, benchmarks['Industrial'])
        
        # Aktuelle Werte berechnen
        annual_breakdown = tco_result['annual_breakdown']
        total_annual = sum(annual_breakdown.values())
        
        actual_maintenance_ratio = annual_breakdown.get('maintenance', 0) / purchase_price
        actual_energy_ratio = annual_breakdown.get('energy', 0) / total_annual if total_annual > 0 else 0
        actual_water_ratio = annual_breakdown.get('water', 0) / total_annual if total_annual > 0 else 0
        actual_tco_multiple = tco_result['cost_summary']['tco_multiple']
        
        # Vergleich erstellen
        comparison = {
            'category_benchmark': category,
            'metrics': {
                'maintenance_ratio': {
                    'actual': actual_maintenance_ratio,
                    'benchmark': benchmark['maintenance_ratio'],
                    'variance': ((actual_maintenance_ratio / benchmark['maintenance_ratio']) - 1) * 100,
                    'status': 'Gut' if actual_maintenance_ratio <= benchmark['maintenance_ratio'] * 1.1 else 'Hoch'
                },
                'energy_ratio': {
                    'actual': actual_energy_ratio,
                    'benchmark': benchmark['energy_ratio'],
                    'variance': ((actual_energy_ratio / benchmark['energy_ratio']) - 1) * 100 if benchmark['energy_ratio'] > 0 else 0,
                    'status': 'Gut' if actual_energy_ratio <= benchmark['energy_ratio'] * 1.1 else 'Hoch'
                },
                'water_ratio': {
                    'actual': actual_water_ratio,
                    'benchmark': benchmark['water_ratio'],
                    'variance': ((actual_water_ratio / benchmark['water_ratio']) - 1) * 100 if benchmark['water_ratio'] > 0 else 0,
                    'status': 'Gut' if actual_water_ratio <= benchmark['water_ratio'] * 1.1 else 'Hoch'
                },
                'tco_multiple': {
                    'actual': actual_tco_multiple,
                    'benchmark': benchmark['tco_multiple'],
                    'variance': ((actual_tco_multiple / benchmark['tco_multiple']) - 1) * 100,
                    'status': 'Gut' if actual_tco_multiple <= benchmark['tco_multiple'] * 1.1 else 'Hoch'
                }
            },
            'overall_rating': 'Exzellent'  # Wird unten berechnet
        }
        
        # Overall Rating berechnen
        high_count = sum(1 for metric in comparison['metrics'].values() if metric['status'] == 'Hoch')
        
        if high_count == 0:
            comparison['overall_rating'] = 'Exzellent'
        elif high_count <= 1:
            comparison['overall_rating'] = 'Gut'
        elif high_count <= 2:
            comparison['overall_rating'] = 'Durchschnittlich'
        else:
            comparison['overall_rating'] = 'Verbesserungsbedarf'
        
        return comparison

if __name__ == "__main__":
    # Test des erweiterten TCO-Calculators
    print("🧪 Teste erweiterten TCO-Calculator...\n")
    
    # Test-Asset (GEA Zentrifuge)
    test_asset = {
        'asset_name': 'SEP-GFA-001',
        'category': 'Citrus',
        'subcategory': 'Citrus Juice Clarification',
        'manufacturer': 'GEA',
        'model': 'GFA 200-30-820',
        'purchase_price': 344261,
        'motor_power_kw': 55,
        'total_power_consumption': 44,
        'water_consumption_ls': 1.2,
        'water_per_ejection': 11,
        'drive_type': 'integrated direct drive',
        'quality_level': 'premium - Level',
        'age_years': 1,
        'usage_pattern': 'Extended (12h/Tag)',
        'location': 'Düsseldorf (HQ)',
        'criticality': 'Hoch'
    }
    
    # TCO berechnen
    calculator = ExtendedTCOCalculator()
    result = calculator.calculate_extended_tco(test_asset, lifetime_years=15)
    
    # Report generieren
    report = calculator.generate_tco_report(result)
    print(report)
    
    print(f"\n📊 Komponenten-Details:")
    for name, cost in result['annual_breakdown'].items():
        if cost > 0:
            confidence = result['confidence_metrics']['component_confidence'][name]
            print(f"- {name}: €{cost:,.0f}/Jahr (Konfidenz: {confidence:.1%})")
    
    # Benchmark-Vergleich
    benchmark = calculator.get_benchmark_comparison(test_asset, result)
    print(f"\n🎯 Benchmark-Vergleich ({benchmark['category_benchmark']}):")
    print(f"- Overall Rating: {benchmark['overall_rating']}")
    for metric_name, metric_data in benchmark['metrics'].items():
        print(f"- {metric_name}: {metric_data['variance']:+.1f}% vs. Benchmark ({metric_data['status']})")
    
    # Optimierungs-Empfehlungen
    recommendations = calculator.get_optimization_recommendations(result)
    print(f"\n💡 Top Optimierungs-Empfehlungen:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['title']} ({rec['priority']})")
        print(f"   💰 Einsparung: €{rec['potential_savings']:,.0f}/Jahr")
        print(f"   ⏱️ {rec['payback_period']}")
    
    # Test mit Energy Agent (falls verfügbar)
    try:
        from energy.energy_agent import EnergyAgent
        print(f"\n🔋 Teste Energy Agent Integration...")
        
        energy_agent = EnergyAgent()
        enhanced_result = calculator.calculate_extended_tco_with_energy_agent(
            test_asset, 
            lifetime_years=15, 
            energy_agent=energy_agent
        )
        
        print(f"✅ Enhanced TCO mit Energy Agent: €{enhanced_result['cost_summary']['total_tco']:,.0f}")
        print(f"⚡ Energy Insights: {len(enhanced_result['energy_insights'].get('recommendations', []))} Optimierungen")
        
        # Vergleiche Standard vs Enhanced
        standard_energy = result['annual_breakdown'].get('energy', 0)
        enhanced_energy = enhanced_result['annual_breakdown'].get('energy', 0)
        
        if abs(standard_energy - enhanced_energy) > 100:  # Significant difference
            print(f"📊 Energiekosten-Unterschied: €{enhanced_energy - standard_energy:,.0f}/Jahr")
            print(f"   Standard: €{standard_energy:,.0f} | Enhanced: €{enhanced_energy:,.0f}")
        else:
            print(f"📊 Energiekosten identisch: €{enhanced_energy:,.0f}/Jahr (Agent verfügbar)")
        
    except ImportError:
        print(f"\n⚠️ Energy Agent nicht verfügbar - Standard TCO-Berechnung verwendet")
    except Exception as e:
        print(f"\n❌ Energy Agent Test fehlgeschlagen: {e}")
    
    print(f"\n🎯 Erweiterte TCO-Analyse erfolgreich!")