import os
import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from fpdf import FPDF

# ---------------------------
# Preis-Funktionen (Platzhalter für API/Excel)
# ---------------------------

def get_energy_price(standort):
    dummy_prices = {
        "Düsseldorf": 0.28,
        "Berlin": 0.27,
        "Mailand": 0.25,
        "Kopenhagen": 0.32,
        "Lyon": 0.24,
    }
    return dummy_prices.get(standort, 0.30)


def get_water_price(standort):
    dummy_prices = {
        "Düsseldorf": 6.0,   # €/m³
        "Berlin": 5.5,
        "Mailand": 5.0,
        "Kopenhagen": 7.0,
        "Lyon": 5.2,
    }
    return dummy_prices.get(standort, 6.0)


def get_wastewater_price(standort):
    dummy_prices = {
        "Düsseldorf": 5.5,   # €/m³ Abwasser
        "Berlin": 6.0,
        "Mailand": 4.8,
        "Kopenhagen": 7.5,
        "Lyon": 5.0,
    }
    return dummy_prices.get(standort, 6.0)


def get_transport_price(standort):
    dummy_prices = {
        "Düsseldorf": 1.0,   # €/t·km (Dummy)
        "Berlin": 0.9,
        "Mailand": 1.2,
        "Kopenhagen": 1.5,
        "Lyon": 1.0,
    }
    return dummy_prices.get(standort, 1.0)

# ---------------------------
# Helper
# ---------------------------

def to_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ["", "-", "n/a", "na", "none"]:
                return default
            # Komma-Decimal nach Punkt wandeln (de-DE CSV/Excel)
            s = s.replace(",", ".")
            return float(s)
        return float(x)
    except:
        return default

def parse_efficiency(x):
    """
    Normalisiert Effizienz:
    - "0,92" -> 0.92
    - 92     -> 0.92 (wenn 1 < e <= 100 → Prozentannahme)
    - sichert gegen 0/negativ und >1 ab.
    """
    e = to_float(x, default=0.9)
    if e > 1.0 and e <= 100.0:
        e = e / 100.0
    # clamp
    if e <= 0:
        e = 0.9
    if e > 1.0:
        e = 1.0
    return max(e, 1e-3)  # nicht zu klein werden lassen


def forecast_cost(base_price, inflation, jahre, annual_consumption):
    """
    Berechnet Gesamtkosten über mehrere Jahre mit Inflation (Zinseszinseffekt).
    base_price: Startpreis (€/kWh oder €/m³)
    inflation: jährliche Rate (0.03 = 3%)
    jahre: Laufzeit in Jahren
    annual_consumption: Verbrauch pro Jahr (z.B. kWh oder m³)
    """
    total = 0.0
    for year in range(1, int(jahre) + 1):
        price = base_price * ((1 + inflation) ** (year - 1))
        total += annual_consumption * price
    return total


# ---------------------------
# TCO Calculation
# ---------------------------


# ---------------------------
# Regeln laden
# ---------------------------


# ---------------------------
# Regeln laden
# ---------------------------
def load_rules(file="rules.json"):
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

rules = load_rules()
# ---------------------------
# Heuristik: Ist eine Spalte wahrscheinlich eine Kosten-Spalte?
# ---------------------------
def is_probably_cost_column(col: str) -> bool:
    col_lower = col.lower()

    # Keywords, die klar für Kosten sprechen
    cost_keywords = [
        "cost", "preis", "price", "fee", "service",
        "maintenance", "energy", "wasser", "transport"
    ]

    # Keywords, die eindeutig Meta/technisch sind
    meta_keywords = [
        "perc", "min", "max", "durchmesser", "diameter",
        "gewicht", "weight", "volumen", "volume",
        "rpm", "speed", "temp", "temperature",
        "druck", "pressure", "feststoff", "solids"
    ]

    # Technische Spalten sofort ausschließen
    if any(mk in col_lower for mk in meta_keywords):
        return False

    # Nur Kosten-relevante Spalten durchlassen
    return any(ck in col_lower for ck in cost_keywords)

# ---------------------------
# TCO Calculation
# ---------------------------

def calculate_tco(row, betriebsstunden, jahre, standort, kundendaten=None):
    strompreis = get_energy_price(standort)
    wasserpreis = get_water_price(standort)
    transportpreis = get_transport_price(standort)

    infl_strom = (kundendaten or {}).get("Strom Inflation", 0.0)
    infl_wasser = (kundendaten or {}).get("Wasser Inflation", 0.0)

    costs = {}

    # --- CAPEX (Listenpreis) ---
    capex = to_float(row.get("Listprice", 0))
    costs["Capex"] = capex

    # --- Energie (Motorleistung) mit Forecast ---
    motor_power = to_float(row.get("SEP_SQLMotorPowerKW", 0))
    if motor_power > 0:
        eff = parse_efficiency(row.get("SEP_SQLMotorEfficiency", 1))
        power = motor_power * 0.8
        annual_kwh = (power * betriebsstunden) / eff
        energy_cost = forecast_cost(strompreis, infl_strom, jahre, annual_kwh)
        costs["Energie"] = energy_cost

    # --- Wasser (Dauerverbrauch) mit Forecast ---
    water = to_float(row.get("SEP_SQLOpWaterls", 0))  # l/s
    if water > 0:
        m3_per_h = (water * 3600) / 1000
        annual_m3 = m3_per_h * betriebsstunden
        wasser_cost = forecast_cost(wasserpreis, infl_wasser, jahre, annual_m3)
        costs["Wasser"] = wasser_cost

    # --- Wasser pro Ejekt mit Forecast ---
    water_eject = to_float(row.get("SEP_SQLOpWaterliteject", 0))  # l/Ejekt
    ejects = to_float(row.get("number of ejection per hour", 0))
    if water_eject > 0 and ejects > 0:
        m3_per_eject = water_eject / 1000
        annual_m3_eject = m3_per_eject * ejects * betriebsstunden
        eject_cost = forecast_cost(wasserpreis, infl_wasser, jahre, annual_m3_eject)
        costs["Eject-Wasser"] = eject_cost

    # --- Servicekosten nach DMR ---
    dmr = to_float(row.get("SEP_SQLDMR", 0))
    if dmr > 0:
        total_hours = betriebsstunden * jahre
        if dmr < 400:
            service_price = 10000
        elif 400 <= dmr <= 700:
            service_price = 15000
        else:
            service_price = 20000
        services_hours = total_hours // 8000
        services_time = jahre // 2
        num_services = max(services_hours, services_time)
        if num_services > 0:
            costs["Service"] = num_services * service_price

    # --- Riemenantrieb (Effizienzverlust) ---
    drive_type = str(row.get("SEP_DriveType", "")).lower()
    if drive_type in ["belt", "riemen", "yes", "ja", "true", "1"]:
        ineffizienz = 0.01 * jahre
        penalty = sum(costs.values()) * ineffizienz
        if penalty > 0:
            costs["Effizienzverlust (Belt)"] = penalty

    # --- Hydrostop Bonus (Pumpenergie-Anteil) ---
    eject_sys = str(row.get("ejection system", "")).lower()
    if eject_sys == "hydrostop" and "Wasser" in costs:
        saving = 0.10 * costs["Wasser"]
        costs["Hydrostop-Einsparung (Pumpenergie)"] = -saving

    # --- Feststoffanteil (Proxy-Verluste) ---
    if kundendaten and "Feststoffanteil" in kundendaten:
        feststoff = to_float(kundendaten["Feststoffanteil"])
        extra_loss = feststoff * 0.001 * betriebsstunden * jahre
        if extra_loss > 0:
            costs["Produktverlust (Fluid)"] = extra_loss

    # --- Transport ---
    weight_tons = to_float(row.get("SEP_SQLTotalWeightKg", 0)) / 1000
    distanz = to_float(kundendaten.get("Distanz_km", 500) if kundendaten else 500)
    if weight_tons > 0 and distanz > 0:
        costs["Transport"] = weight_tons * distanz * transportpreis

    # --- Produktwechsel ---
    volume = to_float(row.get("SEP_SQLBowlVolumeLit", 0))
    if volume > 0 and kundendaten:
        switches = to_float(kundendaten.get("Produktwechsel pro Jahr", 0))
        if switches > 0:
            costs["Produktwechsel"] = volume * switches * jahre

    # ---------------------------
    # Dynamische Regeln aus rules.json
    # ---------------------------
    try:
        with open("rules.json", "r", encoding="utf-8") as f:
            rules = json.load(f)
    except FileNotFoundError:
        rules = {}

    known_input_cols = {
        "Listprice", "Application", "Sub Application", "SEP_SQLLangtyp",
        "SEP_SQLMotorPowerKW", "SEP_SQLMotorEfficiency", "SEP_SQLDMR",
        "SEP_SQLOpWaterls", "SEP_SQLOpWaterliteject", "number of ejection per hour",
        "SEP_SQLTotalWeightKg", "SEP_SQLBowlVolumeLit", "SEP_DriveType"
    }
    already_cost_keys = set(costs.keys())

    for col, val in row.items():
        if col in known_input_cols or col in already_cost_keys:
            continue
        if not isinstance(val, (int, float)) or pd.isna(val) or val == 0:
            continue

        value = to_float(val)
        rule = rules.get(col)

        # --- Meta-Spalte ---
        if isinstance(rule, dict) and rule.get("type") == "meta":
            continue

        # --- Formel als Dict ---
        if isinstance(rule, dict) and "formula" in rule:
            try:
                calc = eval(rule["formula"], {}, {
                    "value": value,
                    "jahre": jahre,
                    "betriebsstunden": betriebsstunden,
                    "wasserpreis": wasserpreis,
                    "strompreis": strompreis,
                    "transportpreis": transportpreis,
                    "infl_strom": infl_strom,
                    "infl_wasser": infl_wasser,
                    "forecast_cost": forecast_cost
                })
                costs[f"Custom: {col}"] = float(calc)
                continue
            except Exception:
                pass

        # --- Formel als String ---
        if isinstance(rule, str):
            try:
                calc = eval(rule, {}, {
                    "value": value,
                    "jahre": jahre,
                    "betriebsstunden": betriebsstunden,
                    "wasserpreis": wasserpreis,
                    "strompreis": strompreis,
                    "transportpreis": transportpreis,
                    "infl_strom": infl_strom,
                    "infl_wasser": infl_wasser,
                    "forecast_cost": forecast_cost
                })
                costs[f"Custom: {col}"] = float(calc)
                continue
            except Exception:
                pass

        # --- Fallback: Standardregel nur bei Kosten-ähnlichen Spalten ---
        if is_probably_cost_column(col):
            est = value * betriebsstunden * jahre
            costs[f"Custom: {col}"] = float(est)

    # --- Gesamtkosten ---
    tco = sum(v for v in costs.values() if isinstance(v, (int, float)))
    costs["TCO"] = tco
    costs["Name"] = row.get("Application", "Unbekannt")
    costs["Maschinen-Nummer"] = row.get("SEP_SQLLangtyp", "-")

    return costs

def yearly_costs_detailed(row, betriebsstunden, jahre, standort, kundendaten):
    """
    Verteilt Capex (Start), Transport (einmalig), Energie/Wasser/Service (über Jahre mit Inflation),
    Sonstiges (alles Positive außerhalb der Kern-Keys) gleichmäßig über die Jahre und Boni/Abzüge (negative Werte) ebenfalls.
    Gibt kumulierte Jahreskosten + Totals zurück.
    """
    n = int(jahre)
    if n <= 0:
        return {"years": [], "cum": [], "totals": {}}

    infl_e = float((kundendaten or {}).get("Strom Inflation", 0.0))
    infl_w = float((kundendaten or {}).get("Wasser Inflation", 0.0))

    capex_total     = to_float(row.get("Capex", 0.0))
    energy_total    = to_float(row.get("Energie", 0.0))
    water_total     = to_float(row.get("Wasser", 0.0))
    service_total   = to_float(row.get("Service", 0.0))
    transport_total = to_float(row.get("Transport", 0.0))

    # alles Positive außerhalb der Kern-Keys sammeln -> Sonstiges
    known_keys = {"Capex", "Energie", "Wasser", "Service", "Transport", "TCO", "Name", "Maschinen-Nummer"}
    sonstiges = sum(
        v for k, v in row.items()
        if isinstance(v, (int, float)) and k not in known_keys and v > 0
    )
    # alle negativen Werte zusammenfassen (Boni/Abzüge)
    bonus = sum(
        v for k, v in row.items()
        if isinstance(v, (int, float)) and v < 0
    )

    # Energie & Wasser geometrisch nach Inflation verteilen
    e_weights = [(1 + infl_e) ** i for i in range(n)]
    energy_by_year = [energy_total * w / sum(e_weights) for w in e_weights] if energy_total > 0 else [0.0]*n

    w_weights = [(1 + infl_w) ** i for i in range(n)]
    water_by_year = [water_total * w / sum(w_weights) for w in w_weights] if water_total > 0 else [0.0]*n

    service_by_year   = [service_total / n] * n if service_total else [0.0]*n
    sonstiges_by_year = [sonstiges / n] * n if sonstiges else [0.0]*n
    transport_by_year = [transport_total] + [0.0]*(n-1) if transport_total else [0.0]*n
    bonus_by_year     = [bonus / n] * n if bonus else [0.0]*n

    # kumulierte Kosten berechnen
    cum = []
    running = 0.0
    for i in range(n):
        if i == 0:
            running += capex_total
        running += (
            transport_by_year[i] + energy_by_year[i] + water_by_year[i] +
            service_by_year[i] + sonstiges_by_year[i] + bonus_by_year[i]
        )
        cum.append(float(running))

    return {
        "years": list(range(1, n+1)),
        "cum": cum,
        "totals": {
            "capex": capex_total,
            "energy": energy_total,
            "water": water_total,
            "service": service_total,
            "transport": transport_total,
            "sonstiges": sonstiges,
            "bonus": bonus,
            "tco_end": cum[-1] if cum else capex_total + transport_total
        }
    }


def yearly_costs(row, betriebsstunden, jahre, standort, kundendaten):
    strompreis = get_energy_price(standort)
    wasserpreis = get_water_price(standort)
    abwasserpreis = get_wastewater_price(standort)

    infl_strom = kundendaten.get("Strom Inflation", 0.03)
    infl_wasser = kundendaten.get("Wasser Inflation", 0.02)

    capex = to_float(row.get("Capex", row.get("Listprice", 0)))
    cum = capex
    yearly = []

    motor_power = to_float(row.get("SEP_SQLMotorPowerKW", 0))
    eff = max(to_float(row.get("SEP_SQLMotorEfficiency", 1), 1), 1e-6)
    power = motor_power * 0.8

    for year in range(1, jahre + 1):
        strompreis_j = strompreis * (1 + infl_strom) ** (year - 1)
        wasserpreis_j = (wasserpreis + abwasserpreis) * (1 + infl_wasser) ** (year - 1)

        energy = power * betriebsstunden * strompreis_j / eff if motor_power > 0 else 0
        water = betriebsstunden * 0.01 * wasserpreis_j  # Dummy analog zu deinem Wasserblock

        cum += energy + water
        yearly.append(cum)

    return yearly

# ---------------------------
# PDF Export
# ---------------------------

def export_pdf_bytes(kundendaten, top3, show_bonuses=False):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "GEA TCO Report", ln=True, align="C")

    # ---------------------------
    # Kundendaten
    # ---------------------------
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Kundendaten:", ln=True)
    for key, value in kundendaten.items():
        if "Inflation" in key:
            pdf.cell(0, 8, f"{key}: {value*100:.1f}%", ln=True)  # Prozent statt Roh-Float
        else:
            pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Top 3 Maschinen (Übersicht):", ln=True)

    # ---------------------------
    # Tabelle Übersicht
    # ---------------------------
    pdf.set_font("Arial", "B", 10)
    headers = ["Maschinen-Nr.", "Capex", "Energie", "Wasser", "Service", "TCO"]
    col_widths = [40, 25, 25, 25, 25, 25]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for _, row in top3.iterrows():
        pdf.cell(40, 10, str(row.get("Maschinen-Nummer","-")), 1)
        pdf.cell(25, 10, f"{to_float(row.get('Capex',0)):,.0f}", 1)
        pdf.cell(25, 10, f"{to_float(row.get('Energie',0)):,.0f}", 1)
        pdf.cell(25, 10, f"{to_float(row.get('Wasser',0)):,.0f}", 1)
        pdf.cell(25, 10, f"{to_float(row.get('Service',0)):,.0f}", 1)
        pdf.cell(25, 10, f"{to_float(row.get('TCO',0)):,.0f}", 1)
        pdf.ln()

    # ---------------------------
    # Detailseiten mit Pie-Charts
    # ---------------------------
    for idx, (_, row) in enumerate(top3.iterrows(), start=1):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Maschine {idx}: {row.get('Maschinen-Nummer','-')}", ln=True)

        # Zahlen holen
        capex_val     = to_float(row.get("Capex", 0))
        transport_val = to_float(row.get("Transport", 0))
        energie_val   = to_float(row.get("Energie", 0))
        wasser_val    = to_float(row.get("Wasser", 0))
        service_val   = to_float(row.get("Service", 0))

        # Sonstiges = alle positiven Restkosten außerhalb der Kern-Keys
        core_keys = {"Capex", "Energie", "Wasser", "Service", "Transport"}
        sonstiges_sum = sum(
            float(v) for k, v in row.items()
            if isinstance(v, (int, float)) and v > 0 and k not in core_keys and k not in ["TCO"]
        )

        # Pie-Chart: Transport in Capex integrieren
        positive_costs_for_pie = {}
        capex_incl_transport = capex_val + transport_val
        if capex_incl_transport > 0:
            positive_costs_for_pie["Capex (inkl. Transport)"] = capex_incl_transport
        if energie_val > 0:
            positive_costs_for_pie["Energie"] = energie_val
        if wasser_val > 0:
            positive_costs_for_pie["Wasser"] = wasser_val
        if service_val > 0:
            positive_costs_for_pie["Service"] = service_val
        if sonstiges_sum > 0:
            positive_costs_for_pie["Sonstiges"] = sonstiges_sum

        negative_costs = {k: v for k, v in row.items()
                          if isinstance(v, (int, float)) and v < 0}

        # Autopct: "Sonstiges" immer labeln, Rest nur >1%
        def autopct_func(all_labels):
            def inner_autopct(p):
                label = all_labels[inner_autopct.idx]
                inner_autopct.idx += 1
                if label == "Sonstiges":
                    return f"{p:.1f}%"
                return f"{p:.1f}%" if p > 1 else ""
            inner_autopct.idx = 0
            return inner_autopct

        # Pie
        if positive_costs_for_pie:
            labels = list(positive_costs_for_pie.keys())
            values = list(positive_costs_for_pie.values())
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct=autopct_func(labels))
            chart_path = f"chart_{idx}.png"
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            pdf.image(chart_path, x=30, w=150)
            os.remove(chart_path)

            # Unter dem Chart: detaillierte Aufschlüsselung inkl. Capex+Transport
            pdf.ln(80)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Capex (Anschaffung): {capex_val:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"Transport (einmalig): {transport_val:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"Capex gesamt (inkl. Transport): {capex_incl_transport:,.0f} EUR", ln=True)
            if energie_val > 0:  pdf.cell(0, 8, f"Energie: {energie_val:,.0f} EUR", ln=True)
            if wasser_val > 0:   pdf.cell(0, 8, f"Wasser: {wasser_val:,.0f} EUR", ln=True)
            if service_val > 0:  pdf.cell(0, 8, f"Service: {service_val:,.0f} EUR", ln=True)
            if sonstiges_sum > 0: pdf.cell(0, 8, f"Sonstiges: {sonstiges_sum:,.0f} EUR", ln=True)

        # Abzüge/Boni optional anzeigen (werden weiterhin mitgerechnet)
        if show_bonuses and negative_costs:
            pdf.ln(5)
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 8, "Abzüge / Boni:", ln=True)
            for lbl, val in negative_costs.items():
                pdf.cell(0, 8, f"- {lbl}: {val:,.0f} EUR", ln=True)

    # ---------------------------
    # Break-Even Chart & Text
    # ---------------------------
    if len(top3) >= 2:
        pdf.add_page()
        n_years = int(kundendaten["Nutzungsdauer (Jahre)"])
        fig, ax = plt.subplots()

        endcosts = []

        for _, row in top3.iterrows():
            breakdown = yearly_costs_detailed(
                row,
                kundendaten["Betriebsstunden"],
                n_years,
                kundendaten["Standort"],
                kundendaten
            )
            maschinen_nr = row.get("Maschinen-Nummer","-")
            cum_values = [float(x) for x in breakdown["cum"]]

            line, = ax.plot(breakdown["years"], cum_values, label=f"Maschine {maschinen_nr}")

            totals = breakdown["totals"]
            endcost = totals['tco_end']
            endcosts.append((maschinen_nr, endcost))

            ax.annotate(f"{endcost:,.0f} EUR",
                        xy=(breakdown["years"][-1], cum_values[-1]),
                        xytext=(5,0), textcoords="offset points",
                        fontsize=8, color=line.get_color())

            # Text im PDF
            capex_only   = totals['capex']
            transport    = totals['transport']
            capex_plus_t = capex_only + transport

            pdf.ln(5)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"Maschine {maschinen_nr}", ln=True)
            pdf.cell(0, 8, f"  Startkosten (Capex): {capex_only:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Transport (einmalig): {transport:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Capex gesamt (inkl. Transport): {capex_plus_t:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Energie über {n_years} Jahre: {totals['energy']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Wasser über {n_years} Jahre: {totals['water']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Service über {n_years} Jahre: {totals['service']:,.0f} EUR", ln=True)
            if abs(totals.get("sonstiges", 0.0)) > 1e-6:
                pdf.cell(0, 8, f"  Sonstiges: {totals['sonstiges']:,.0f} EUR", ln=True)
            if show_bonuses and abs(totals.get("bonus", 0.0)) > 1e-6:
                pdf.cell(0, 8, f"  Abzüge/Boni: {totals['bonus']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Gesamtkosten nach {n_years} Jahren: {endcost:,.0f} EUR", ln=True)

        ax.set_xlabel("Jahre")
        ax.set_ylabel("Kumulierte Kosten (EUR)")
        ax.ticklabel_format(style="plain", axis="y")
        ax.set_ylim(0, None)
        ax.legend()
        plt.title("Break-Even Vergleich (3 Maschinen)")
        img_path = "temp_breakeven.png"
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()
        pdf.image(img_path, x=30, w=150)
        os.remove(img_path)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Endkosten aus dem Diagramm:", ln=True)
        for maschinen_nr, endcost in endcosts:
            pdf.cell(0, 8, f"- Maschine {maschinen_nr}: {endcost:,.0f} EUR", ln=True)

    # ---------------------------
    # Bytes statt Datei zurückgeben
    # ---------------------------
    out = pdf.output(dest='S')
    if isinstance(out, str):
        pdf_bytes = out.encode('latin-1')
    else:
        pdf_bytes = bytes(out)   # sicherstellen, dass es wirklich bytes sind
    return pdf_bytes, "GEA_TCO_Report.pdf"

# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.title("GEA TCO Insight Tool (Smart)")

    uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])
    if uploaded_file:
        with open("uploaded.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        xls = pd.ExcelFile("uploaded.xlsx")
    elif os.path.exists("uploaded.xlsx"):
        xls = pd.ExcelFile("uploaded.xlsx")
    else:
        st.warning("Bitte eine Excel hochladen.")
        return

    sheet_name = st.selectbox("Tabelle auswählen", xls.sheet_names)
    df = xls.parse(sheet_name)
    st.dataframe(df.head())

    machine_types = df["Application"].dropna().unique()
    selected_type = st.selectbox("Maschinentyp auswählen", machine_types)

    sub_types = df[df["Application"] == selected_type]["Sub Application"].dropna().unique()
    selected_sub = None
    if len(sub_types) > 0:
        selected_sub = st.selectbox("Sub Application (optional)", ["Alle"] + list(sub_types))

    st.subheader("Kundendaten")
    kunde_name = st.text_input("Kunde / Projektname")
    standort = st.selectbox("Standort", ["Düsseldorf", "Berlin", "Mailand", "Kopenhagen", "Lyon"])
    feststoffanteil = st.number_input("Feststoffanteil (%)", 0.0, 100.0, 5.0, 0.5)
    betriebsstunden = st.number_input("Betriebsstunden pro Jahr", 100, 8760, 4000, 100)
    nutzungsdauer = st.number_input("Nutzungsdauer (Jahre)", 1, 100000, 10)
    budget = st.number_input("Budget (EUR, optional)", 0, value=0)  # optional

    # ---------------------------
    # Forecast-Parameter
    # ---------------------------
    st.subheader("Forecast-Parameter")
    strom_inflation = st.number_input("Strompreissteigerung (% pro Jahr)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    wasser_inflation = st.number_input("Wasser-/Abwasserpreissteigerung (% pro Jahr)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

    kundendaten = {
        "Kunde": kunde_name,
        "Standort": standort,
        "Feststoffanteil": feststoffanteil,
        "Betriebsstunden": betriebsstunden,
        "Nutzungsdauer (Jahre)": nutzungsdauer,
        "Budget": budget,
        "Maschinentyp": selected_type,
        "Sub Application": selected_sub if selected_sub and selected_sub != "Alle" else "Alle",
        "Strom Inflation": strom_inflation / 100.0,
        "Wasser Inflation": wasser_inflation / 100.0,
    }

    if st.button("Maschinen berechnen"):
        if selected_sub and selected_sub != "Alle":
            filtered_df = df[(df["Application"] == selected_type) & (df["Sub Application"] == selected_sub)]
        else:
            filtered_df = df[df["Application"] == selected_type]

        results = []
        for _, row in filtered_df.iterrows():
            res = calculate_tco(row, betriebsstunden, nutzungsdauer, standort, kundendaten)
            results.append(res)

        res_df = pd.DataFrame(results).sort_values("TCO").reset_index(drop=True)

        # Budgetfilter optional
        if budget and budget > 0:
            res_df = res_df[res_df["TCO"] <= budget]

        st.session_state["results_df"] = res_df

    if "results_df" in st.session_state:
        res_df = st.session_state["results_df"]
        if res_df.empty:
            st.warning("Keine Maschine erfüllt die Kriterien (Budgetfilter aktiv?).")
        else:
            top3 = res_df.head(3)

            st.subheader("Top 3 Maschinen")
            st.dataframe(top3)

            # Checkbox immer sichtbar
            show_bonuses = st.checkbox(
                "Boni im PDF anzeigen?",
                value=st.session_state.get("show_bonuses", False),
                key="show_bonuses"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("PDF erstellen", key="btn_build_pdf"):
                    pdf_bytes, fname = export_pdf_bytes(kundendaten, top3, show_bonuses=show_bonuses)
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["pdf_filename"] = fname

            with col2:
                if "pdf_bytes" in st.session_state:
                    st.download_button(
                        "Download PDF",
                        data=st.session_state["pdf_bytes"],
                        file_name=st.session_state.get("pdf_filename", "GEA_TCO_Report.pdf"),
                        mime="application/pdf",
                        key="btn_download_pdf"
                    )

if __name__ == "__main__":
    main()
