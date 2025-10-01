import os
import streamlit as st
import pandas as pd
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
        if isinstance(x, str) and x.strip().lower() in ["", "-", "n/a", "na", "none"]:
            return default
        return float(x)
    except:
        return default

# ---------------------------
# TCO Calculation
# ---------------------------

def calculate_tco(row, betriebsstunden, jahre, standort, kundendaten=None):
    strompreis = get_energy_price(standort)
    wasserpreis = get_water_price(standort)
    abwasserpreis = get_wastewater_price(standort)
    transportpreis = get_transport_price(standort)

    # Forecast-Inflationsraten (Defaults: Strom ~3%, Wasser ~2% -> im UI auf %/100 setzen)
    infl_strom = (kundendaten or {}).get("Strom Inflation", 0.0)
    infl_wasser = (kundendaten or {}).get("Wasser Inflation", 0.0)

    # Duty-Cycle für kontinuierlichen Wasserfluss (0..1). Falls Nutzer % eingibt, konvertieren.
    duty = to_float((kundendaten or {}).get("Wasser Duty Cycle", 0.2), 0.2)
    if duty > 1.0:
        duty = duty / 100.0
    duty = max(0.0, min(duty, 1.0))

    # Optional: kontinuierlicher Wasserfluss ganz deaktivieren (nur Ejektionswasser)
    cont_flow_enabled = bool((kundendaten or {}).get("Dauerfluss aktiv", True))

    costs = {}

    # ---------------------------
    # CAPEX
    # ---------------------------
    capex = to_float(row.get("Listprice", 0))
    costs["Capex"] = capex

    # ---------------------------
    # ENERGIE mit Forecast
    # ---------------------------
    motor_power = to_float(row.get("SEP_SQLMotorPowerKW", row.get("power consumption TOTAL [kW]", 0)))
    energy_cost = 0.0
    if motor_power > 0:
        eff = to_float(row.get("SEP_SQLMotorEfficiency", 1), 1)
        if eff <= 0: eff = 1
        power = motor_power * 0.8  # 80% Last
        for year in range(jahre):
            strompreis_jahr = strompreis * (1 + infl_strom) ** year
            energy_cost += power * betriebsstunden * strompreis_jahr / eff
        costs["Energie"] = energy_cost

    # ---------------------------
    # WASSER + ABWASSER + PUMPENERGIE (mit Forecast & Duty-Cycle)
    # ---------------------------
    water_total = 0.0
    water_l = to_float(row.get("SEP_SQLOpWaterls", 0))  # l/s oder l/h
    unit = (kundendaten or {}).get("Wasser Einheit", "l/s")
    pump_eff = max(min(to_float((kundendaten or {}).get("Pumpenwirkungsgrad", 0.6), 0.6), 0.95), 0.05)

    # Durchfluss-Basis in m³/h und m³/s
    m3_per_h_base = 0.0
    Q_m3_s_base = 0.0
    if water_l > 0:
        if unit == "l/s":
            m3_per_h_base = (water_l * 3600.0) / 1000.0
            Q_m3_s_base = water_l / 1000.0
        else:  # l/h
            m3_per_h_base = water_l / 1000.0
            Q_m3_s_base = (water_l / 3600.0) / 1000.0

    # Duty-Cycle anwenden (falls Dauerfluss aktiv)
    m3_per_h_cont = m3_per_h_base * duty if cont_flow_enabled else 0.0
    Q_m3_s_cont = Q_m3_s_base * duty if cont_flow_enabled else 0.0

    # Druckquelle (Excel oder manuell)
    if (kundendaten or {}).get("Excel-Druck verwenden", True):
        pressure_bar = to_float(row.get("SEP_SQLOpWaterSupplyBar", 0))
        if pressure_bar == 0:
            pressure_bar = to_float((kundendaten or {}).get("Manueller Druck (bar)", 2.5), 2.5)
    else:
        pressure_bar = to_float((kundendaten or {}).get("Manueller Druck (bar)", 2.5), 2.5)

    # Pumpenleistung nur für den (ggf. gedrosselten) Dauerfluss
    pump_power_kw_cont = 0.0
    if Q_m3_s_cont > 0 and pressure_bar > 0:
        delta_p_pa = pressure_bar * 1e5  # 1 bar = 1e5 Pa
        pump_power_kw_cont = (Q_m3_s_cont * delta_p_pa) / (pump_eff * 1000.0)  # W→kW

    # Jährliche Aufsummierung: Volumen + Pumpenergie (kont. Anteil)
    for year in range(jahre):
        wasserpreis_mix = (wasserpreis + abwasserpreis) * (1 + infl_wasser) ** year
        # Volumengebühren (nur wenn Dauerfluss aktiv)
        if m3_per_h_cont > 0:
            water_total += m3_per_h_cont * betriebsstunden * wasserpreis_mix
        # Pumpenergie für Dauerfluss
        if pump_power_kw_cont > 0:
            strompreis_jahr = strompreis * (1 + infl_strom) ** year
            pump_energy_cost_year = pump_power_kw_cont * betriebsstunden * strompreis_jahr
            # Hydrostop → 10% weniger Pumpenergie (nur auf Pumpenergie!)
            eject_sys = str(row.get("ejection system", "")).lower()
            if eject_sys == "hydrostop":
                saving = 0.10 * pump_energy_cost_year
                pump_energy_cost_year -= saving
                costs["Hydrostop-Einsparung (Pumpenergie)"] = costs.get("Hydrostop-Einsparung (Pumpenergie)", 0) - saving
            water_total += pump_energy_cost_year

    # Ejektionswasser (unabhängig vom Duty-Cycle, da über Ejektionen definiert)
    water_eject_l = to_float(row.get("SEP_SQLOpWaterliteject", 0))  # l/Ejekt
    ejects_h = to_float(row.get("number of ejection per hour", 0))
    if water_eject_l > 0 and ejects_h > 0:
        m3_per_eject = water_eject_l / 1000.0
        for year in range(jahre):
            wasserpreis_mix = (wasserpreis + abwasserpreis) * (1 + infl_wasser) ** year
            water_total += m3_per_eject * ejects_h * betriebsstunden * wasserpreis_mix

    # Optional: Plausibilisierung vs. Energie (falls gewünscht)
    if (kundendaten or {}).get("Wasser plausibilisieren", False) and energy_cost > 0:
        lower_ratio = (kundendaten or {}).get("Wasser Min Ratio", 0.6)
        upper_ratio = (kundendaten or {}).get("Wasser Max Ratio", 1.2)
        water_total = max(min(water_total, energy_cost * upper_ratio), energy_cost * lower_ratio)

    if water_total > 0:
        costs["Wasser"] = water_total

    # ---------------------------
    # SERVICE (vereinfachte Logik)
    # ---------------------------
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

    # ---------------------------
    # EFFIZIENZVERLUST RIEMEN
    # ---------------------------
    drive_type = str(row.get("SEP_DriveType", "")).lower()
    if drive_type in ["belt", "riemen", "yes", "ja", "true", "1"]:
        ineffizienz = 0.01 * jahre
        penalty = sum(v for k, v in costs.items() if isinstance(v, (int, float))) * ineffizienz
        if penalty > 0:
            costs["Effizienzverlust (Belt)"] = penalty

    # ---------------------------
    # FESTSTOFFVERLUST
    # ---------------------------
    if kundendaten and "Feststoffanteil" in kundendaten:
        feststoff = to_float(kundendaten["Feststoffanteil"])
        extra_loss = feststoff * 0.001 * betriebsstunden * jahre
        if extra_loss > 0:
            costs["Produktverlust (Fluid)"] = extra_loss

    # ---------------------------
    # TRANSPORT
    # ---------------------------
    weight_tons = to_float(row.get("SEP_SQLTotalWeightKg", 0)) / 1000.0
    distanz = to_float((kundendaten or {}).get("Distanz_km", 500))
    if weight_tons > 0 and distanz > 0:
        costs["Transport"] = weight_tons * distanz * transportpreis

    # ---------------------------
    # PRODUKTWECHSEL
    # ---------------------------
    volume = to_float(row.get("SEP_SQLBowlVolumeLit", 0))
    if volume > 0 and kundendaten:
        switches = to_float(kundendaten.get("Produktwechsel pro Jahr", 0))
        if switches > 0:
            costs["Produktwechsel"] = volume * switches * jahre

    # ---------------------------
    # GESAMT
    # ---------------------------
    tco = sum(v for k, v in costs.items() if isinstance(v, (int, float)))
    costs["TCO"] = tco
    costs["Name"] = row.get("Application", "Unbekannt")
    costs["Maschinen-Nummer"] = str(row.get("SEP_SQLLangtyp", "-"))

    if kundendaten is not None:
        kundendaten["Maschinen-Nummer"] = costs["Maschinen-Nummer"]

    return costs

def yearly_costs_detailed(row, betriebsstunden, jahre, standort, kundendaten):
    """
    Nutzt die aggregierten Kosten (Capex, Energie, Wasser, Service, Transport usw.)
    und verteilt sie jahresweise mit Inflationseffekten.
    Gibt Float-Liste 'cum' zurück, die bei Capex startet und bei TCO endet.
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

    # andere positive und negative Blöcke einsammeln
    known = {"Capex", "Energie", "Wasser", "Service", "Transport", "TCO", "Name", "Maschinen-Nummer"}
    other_pos = sum(v for k, v in row.items() if isinstance(v, (int, float)) and k not in known and v > 0)
    negatives = sum(v for k, v in row.items() if isinstance(v, (int, float)) and v < 0)

    # Energie & Wasser geometrisch nach Inflation
    w_e = [(1 + infl_e) ** i for i in range(n)]
    energy_by_year = [energy_total * w / sum(w_e) for w in w_e] if energy_total > 0 else [0.0]*n

    w_w = [(1 + infl_w) ** i for i in range(n)]
    water_by_year = [water_total * w / sum(w_w) for w in w_w] if water_total > 0 else [0.0]*n

    service_by_year   = [service_total / n] * n if service_total else [0.0]*n
    other_by_year     = [other_pos / n] * n if other_pos else [0.0]*n
    transport_by_year = [transport_total] + [0.0]*(n-1) if transport_total else [0.0]*n
    bonus_by_year     = [negatives / n] * n if negatives else [0.0]*n

    # kumuliert aufbauen
    cum = []
    running = 0.0
    for i in range(n):
        if i == 0:
            running += capex_total
        running += (transport_by_year[i] + energy_by_year[i] + water_by_year[i] +
                    service_by_year[i] + other_by_year[i] + bonus_by_year[i])
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
            "other": other_pos,
            "bonus": negatives,
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

def export_pdf(kundendaten, top3):
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

        positive_costs = {k: v for k, v in row.items()
                          if isinstance(v, (int, float)) and v > 0 and k not in ["TCO"]}
        negative_costs = {k: v for k, v in row.items()
                          if isinstance(v, (int, float)) and v < 0}

        if positive_costs:
            labels = list(positive_costs.keys())
            values = list(positive_costs.values())
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%")
            chart_path = f"chart_{idx}.png"
            plt.savefig(chart_path)
            plt.close()
            pdf.image(chart_path, x=30, w=150)
            os.remove(chart_path)

            pdf.ln(80)
            pdf.set_font("Arial", "", 12)
            for lbl, val in positive_costs.items():
                pdf.cell(0, 8, f"{lbl}: {val:,.0f} EUR", ln=True)

        if negative_costs:
            pdf.ln(5)
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 8, "Abzüge / Boni:", ln=True)
            for lbl, val in negative_costs.items():
                pdf.cell(0, 8, f"- {lbl}: {val:,.0f} EUR", ln=True)

    # ---------------------------
    # Break-Even Chart (3 Linien, Endkosten annotiert + im Text)
    # ---------------------------
    if len(top3) >= 2:
        pdf.add_page()
        n_years = int(kundendaten["Nutzungsdauer (Jahre)"])
        fig, ax = plt.subplots()

        endcosts = []  # sammeln für Text unter dem Diagramm

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

            # Endwert im Diagramm annotieren (rechts neben der Linie)
            ax.annotate(f"{endcost:,.0f} EUR",
                        xy=(breakdown["years"][-1], cum_values[-1]),
                        xytext=(5,0), textcoords="offset points",
                        fontsize=8, color=line.get_color())

            # Detaillierte Werte ins PDF schreiben
            pdf.ln(5)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"Maschine {maschinen_nr}", ln=True)
            pdf.cell(0, 8, f"  Startkosten (Capex): {totals['capex']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Transport (einmalig): {totals['transport']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Energie über {n_years} Jahre: {totals['energy']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Wasser über {n_years} Jahre: {totals['water']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Service über {n_years} Jahre: {totals['service']:,.0f} EUR", ln=True)
            if abs(totals.get("other", 0.0)) > 1e-6:
                pdf.cell(0, 8, f"  Weitere Kosten: {totals['other']:,.0f} EUR", ln=True)
            if abs(totals.get("bonus", 0.0)) > 1e-6:
                pdf.cell(0, 8, f"  Abzüge/Boni: {totals['bonus']:,.0f} EUR", ln=True)
            pdf.cell(0, 8, f"  Gesamtkosten nach {n_years} Jahren: {endcost:,.0f} EUR", ln=True)

        # Achseneinstellungen
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

        # Endkosten zusätzlich gesammelt unten schreiben
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Endkosten aus dem Diagramm:", ln=True)
        for maschinen_nr, endcost in endcosts:
            pdf.cell(0, 8, f"- Maschine {maschinen_nr}: {endcost:,.0f} EUR", ln=True)

    output_path = "GEA_TCO_Report.pdf"
    pdf.output(output_path)
    return output_path

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
    nutzungsdauer = st.number_input("Nutzungsdauer (Jahre)", 1, 100000, 10)  # ⚡ ohne Limit
    budget = st.number_input("Budget (EUR)", 0, value=500000)

    # ---------------------------
    # Forecast-Parameter (% pro Jahr statt Slider)
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
        st.session_state["results_df"] = res_df

    if "results_df" in st.session_state:
        res_df = st.session_state["results_df"]
        top3 = res_df.head(3)

        st.subheader("Top 3 Maschinen")
        st.dataframe(top3)

        if st.button("PDF erstellen"):
            pdf_path = export_pdf(kundendaten, top3)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_path)

if __name__ == "__main__":
    main()
