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

    costs = {}

    # Capex (Listenpreis)
    capex = to_float(row.get("Listprice", 0))
    costs["Capex"] = capex

    # Motorleistung → Energieverbrauch
    motor_power = to_float(row.get("SEP_SQLMotorPowerKW", row.get("power consumption TOTAL [kW]", 0)))
    energy_cost = 0.0
    if motor_power > 0:
        eff = to_float(row.get("SEP_SQLMotorEfficiency", 1), 1)
        if eff <= 0:
            eff = 1
        power = motor_power * 0.8
        energy_cost = power * betriebsstunden * jahre * strompreis / eff
        costs["Energie"] = energy_cost

    # --- WASSERKOSTEN: Volumen + Abwasser + Pumpenergie ---
    water_total = 0.0
    water_l = to_float(row.get("SEP_SQLOpWaterls", 0))
    unit = (kundendaten or {}).get("Wasser Einheit", "l/s")
    pump_eff = max(min(to_float((kundendaten or {}).get("Pumpenwirkungsgrad", 0.6), 0.6), 0.95), 0.05)

    if water_l > 0:
        if unit == "l/s":
            m3_per_h = (water_l * 3600.0) / 1000.0
            Q_m3_s = water_l / 1000.0
        else:  # l/h
            m3_per_h = water_l / 1000.0
            Q_m3_s = (water_l / 3600.0) / 1000.0

        # Volumen-Kosten (Wasser + Abwasser)
        water_total += m3_per_h * betriebsstunden * jahre * (wasserpreis + abwasserpreis)

        # Pumpkosten (über Druck)
        if (kundendaten or {}).get("Excel-Druck verwenden", True):
            pressure_bar = to_float(row.get("SEP_SQLOpWaterSupplyBar", 0))
            if pressure_bar == 0:
                pressure_bar = to_float((kundendaten or {}).get("Manueller Druck (bar)", 2.5), 2.5)
        else:
            pressure_bar = to_float((kundendaten or {}).get("Manueller Druck (bar)", 2.5), 2.5)

        pump_energy_cost = 0.0
        if Q_m3_s > 0 and pressure_bar > 0:
            delta_p_pa = pressure_bar * 1e5
            pump_power_kw = (Q_m3_s * delta_p_pa) / (pump_eff * 1000.0)
            pump_energy_cost = pump_power_kw * betriebsstunden * jahre * strompreis

            # Hydrostop → 10% weniger Pumpenergie
            eject_sys = str(row.get("ejection system", "")).lower()
            if eject_sys == "hydrostop":
                saving = 0.10 * pump_energy_cost
                pump_energy_cost -= saving
                costs["Hydrostop-Einsparung (Pumpenergie)"] = -saving

            water_total += pump_energy_cost

    # Wasser pro Ejekt
    water_eject_l = to_float(row.get("SEP_SQLOpWaterliteject", 0))
    ejects_h = to_float(row.get("number of ejection per hour", 0))
    if water_eject_l > 0 and ejects_h > 0:
        m3_per_eject = water_eject_l / 1000.0
        water_total += m3_per_eject * ejects_h * betriebsstunden * jahre * (wasserpreis + abwasserpreis)

    # Plausibilisierung: Wasser vs. Energie
    if (kundendaten or {}).get("Wasser plausibilisieren", False) and energy_cost > 0:
        lower_ratio = (kundendaten or {}).get("Wasser Min Ratio", 0.6)
        upper_ratio = (kundendaten or {}).get("Wasser Max Ratio", 1.2)
        water_total = max(min(water_total, energy_cost * upper_ratio), energy_cost * lower_ratio)

    if water_total > 0:
        costs["Wasser"] = water_total

    # Servicekosten (vereinfachte Annahme)
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

    # Effizienzverluste Riemen
    drive_type = str(row.get("SEP_DriveType", "")).lower()
    if drive_type in ["belt", "riemen", "yes", "ja", "true", "1"]:
        ineffizienz = 0.01 * jahre
        penalty = sum(v for k, v in costs.items() if isinstance(v, (int, float))) * ineffizienz
        if penalty > 0:
            costs["Effizienzverlust (Belt)"] = penalty

    # Feststoffanteil
    if kundendaten and "Feststoffanteil" in kundendaten:
        feststoff = to_float(kundendaten["Feststoffanteil"])
        extra_loss = feststoff * 0.001 * betriebsstunden * jahre
        if extra_loss > 0:
            costs["Produktverlust (Fluid)"] = extra_loss

    # Transport
    weight_tons = to_float(row.get("SEP_SQLTotalWeightKg", 0)) / 1000.0
    distanz = to_float((kundendaten or {}).get("Distanz_km", 500))
    if weight_tons > 0 and distanz > 0:
        costs["Transport"] = weight_tons * distanz * transportpreis

    # Produktwechsel
    volume = to_float(row.get("SEP_SQLBowlVolumeLit", 0))
    if volume > 0 and kundendaten:
        switches = to_float(kundendaten.get("Produktwechsel pro Jahr", 0))
        if switches > 0:
            costs["Produktwechsel"] = volume * switches * jahre

    # Gesamtkosten
    tco = sum(v for k, v in costs.items() if isinstance(v, (int, float)))
    costs["TCO"] = tco

    # Infos
    costs["Name"] = row.get("Application", "Unbekannt")
    costs["Maschinen-Nummer"] = str(row.get("SEP_SQLLangtyp", "-"))

    if kundendaten is not None:
        kundendaten["Maschinen-Nummer"] = costs["Maschinen-Nummer"]

    return costs

# ---------------------------
# PDF Export
# ---------------------------

def export_pdf(kundendaten, top3):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "GEA TCO Report", ln=True, align="C")

    # Kundendaten
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Kundendaten:", ln=True)
    for key, value in kundendaten.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(3)
    if len(top3) > 0:
        ids = [str(x) for x in top3["Maschinen-Nummer"].tolist() if pd.notna(x)]
        if ids:
            pdf.cell(0, 8, f"Maschinen-Nummern (Top Auswahl): {', '.join(ids)}", ln=True)
    pdf.cell(0, 8, f"Sub Application: {kundendaten.get('Sub Application','Alle')}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Top 3 Maschinen (Übersicht):", ln=True)

    # Tabelle
    pdf.set_font("Arial", "B", 10)
    headers = ["Maschinen-Nr.", "Capex", "Energie", "Wasser", "Service", "TCO"]
    col_widths = [40, 25, 25, 25, 25, 25]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for _, row in top3.iterrows():
        pdf.cell(40, 10, str(row.get("Maschinen-Nummer","-")), 1)
        pdf.cell(25, 10, f"{row.get('Capex',0):.0f}", 1)
        pdf.cell(25, 10, f"{row.get('Energie',0):.0f}", 1)
        pdf.cell(25, 10, f"{row.get('Wasser',0):.0f}", 1)
        pdf.cell(25, 10, f"{row.get('Service',0):.0f}", 1)
        pdf.cell(25, 10, f"{row['TCO']:.0f}", 1)
        pdf.ln()

    # Detailseiten
    for idx, (_, row) in enumerate(top3.iterrows(), start=1):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Maschine {idx}: {row.get('Maschinen-Nummer','-')}", ln=True)

        positive_costs = {k: v for k, v in row.items() if isinstance(v, (int, float)) and v > 0 and k not in ["TCO"]}
        negative_costs = {k: v for k, v in row.items() if isinstance(v, (int, float)) and v < 0}

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

    # Break-Even
    if len(top3) >= 2:
        pdf.add_page()
        jahre = list(range(1, int(kundendaten["Nutzungsdauer (Jahre)"]) + 1))
        fig, ax = plt.subplots()

        for _, row in top3.iterrows():
            yearly_opex = sum(v for k, v in row.items() if k not in ["Maschinen-Nummer", "Name", "Capex", "TCO"]
                              and isinstance(v, (int, float)) and v > 0)
            kosten = [row.get("Capex", 0) + yearly_opex * j for j in jahre]
            ax.plot(jahre, kosten, label=str(row.get("Maschinen-Nummer","-")))

        ax.set_xlabel("Jahre")
        ax.set_ylabel("Kumulierte Kosten (€)")
        ax.legend()
        plt.title("Break-Even Vergleich")
        img_path = "temp_breakeven.png"
        plt.savefig(img_path)
        plt.close()
        pdf.image(img_path, x=30, w=150)
        os.remove(img_path)

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

    # Auswahl
    machine_types = df["Application"].dropna().unique()
    selected_type = st.selectbox("Maschinentyp auswählen", machine_types)

    sub_types = df[df["Application"] == selected_type]["Sub Application"].dropna().unique()
    selected_sub = None
    if len(sub_types) > 0:
        selected_sub = st.selectbox("Sub Application (optional)", ["Alle"] + list(sub_types))

    # Kundendaten
    st.subheader("Kundendaten")
    kunde_name = st.text_input("Kunde / Projektname")
    standort = st.selectbox("Standort", ["Düsseldorf", "Berlin", "Mailand", "Kopenhagen", "Lyon"])
    feststoffanteil = st.number_input("Feststoffanteil (%)", 0.0, 100.0, 5.0, 0.5)
    betriebsstunden = st.number_input("Betriebsstunden pro Jahr", 100, 8760, 4000, 100)
    nutzungsdauer = st.number_input("Nutzungsdauer (Jahre)", 1, 30, 10)
    budget = st.number_input("Budget (€)", 0, value=500000)
    distanz_km = st.number_input("Transportdistanz (km)", 0, value=500)
    produktwechsel = st.number_input("Produktwechsel pro Jahr", 0, value=0)

    # Wasser-Einstellungen
    st.subheader("Wasserkosten-Einstellungen")
    water_unit = st.selectbox("Einheit Wasserverbrauch", ["l/s", "l/h"], index=0)
    pump_eff = st.number_input("Pumpenwirkungsgrad η", 0.1, 1.0, 0.6, 0.05)
    use_excel_pressure = st.checkbox("Druck aus Excel verwenden (SEP_SQLOpWaterSupplyBar)", True)
    manual_pressure = st.number_input("Manueller Wasserdruck (bar)", 0.0, 20.0, 2.5, 0.1)
    ww_price = st.number_input("Abwasserpreis (€/m³)", 0.0, 20.0, float(get_wastewater_price(standort)), 0.1)
    water_plaus = st.checkbox("Wasserkosten an Stromkosten plausibilisieren (~0.6×–1.2×)", True)
    min_ratio = st.slider("Min. Verhältnis zu Energie", 0.1, 1.0, 0.6, 0.05)
    max_ratio = st.slider("Max. Verhältnis zu Energie", 0.5, 2.0, 1.2, 0.05)

    kundendaten = {
        "Kunde": kunde_name,
        "Standort": standort,
        "Feststoffanteil": feststoffanteil,
        "Betriebsstunden": betriebsstunden,
        "Nutzungsdauer (Jahre)": nutzungsdauer,
        "Budget": budget,
        "Distanz_km": distanz_km,
        "Produktwechsel pro Jahr": produktwechsel,
        "Maschinentyp": selected_type,
        "Sub Application": selected_sub if selected_sub and selected_sub != "Alle" else "Alle",
        "Wasser Einheit": water_unit,
        "Pumpenwirkungsgrad": float(pump_eff),
        "Excel-Druck verwenden": bool(use_excel_pressure),
        "Manueller Druck (bar)": float(manual_pressure),
        "Abwasserpreis": float(ww_price),
        "Wasser plausibilisieren": water_plaus,
        "Wasser Min Ratio": float(min_ratio),
        "Wasser Max Ratio": float(max_ratio),
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

        for _, machine in top3.iterrows():
            st.subheader(f"Analyse für Maschine {machine.get('Maschinen-Nummer','-')}")
            pos_costs = {k: v for k, v in machine.items() if isinstance(v, (int, float)) and v > 0 and k not in ["TCO"]}
            neg_costs = {k: v for k, v in machine.items() if isinstance(v, (int, float)) and v < 0}

            if pos_costs:
                labels = list(pos_costs.keys())
                values = list(pos_costs.values())
                fig, ax = plt.subplots()
                ax.pie(values, labels=labels, autopct="%1.1f%%")
                st.pyplot(fig)
                st.write("**Kostenaufstellung:**")
                for lbl, val in pos_costs.items():
                    st.write(f"- {lbl}: {val:,.0f} EUR")

            if neg_costs:
                st.write("**Abzüge / Boni:**")
                for lbl, val in neg_costs.items():
                    st.write(f"- {lbl}: {val:,.0f} EUR")

        if st.button("PDF erstellen"):
            pdf_path = export_pdf(kundendaten, top3)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_path)


if __name__ == "__main__":
    main()
