import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# 0. SEITEN-KONFIGURATION & CUSTOM CSS DESIGN
# =====================================================================
st.set_page_config(
    page_title="In-silico Glukose-Insulin-Dynamik | Menstruationszyklus & KOK",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Gesamter Seitenhintergrund */
    .stApp {
        background: linear-gradient(180deg, #fff9fb 0%, #ffffff 100%);
        color: #3a1d28;
    }

    /* Haupt-Header */
    .hero-header {
        background: #ffffff;
        border: 1px solid #f8bbd0;
        border-radius: 18px;
        padding: 22px 28px;
        box-shadow: 0 8px 24px rgba(233, 30, 99, 0.10);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ad1457;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #6d4c58;
        margin: 0;
    }

    /* Weisse Steuerungs-Box mit rosa Schatten */
    .control-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #f8bbd0;
        box-shadow: 0 8px 22px rgba(233, 30, 99, 0.18);
        margin-bottom: 18px;
    }
    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ad1457;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Hellrosa Messwerte-Box mit Schatten */
    .metrics-card {
        background-color: #fff0f5;
        border-radius: 16px;
        padding: 20px 22px;
        border: 1px solid #f48fb1;
        box-shadow: 0 8px 22px rgba(233, 30, 99, 0.20);
        margin-top: 6px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px dashed #f8bbd0;
    }
    .metric-row:last-of-type {
        border-bottom: none;
    }
    .metric-label {
        font-size: 0.92rem;
        color: #6d4c58;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #880e4f;
    }
    .si-highlight {
        background: #ffffff;
        border: 1px solid #f48fb1;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 10px 0 14px 0;
        text-align: center;
        box-shadow: 0 3px 10px rgba(216, 27, 96, 0.08);
    }
    .si-highlight-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #ad1457;
        font-weight: 600;
    }
    .si-highlight-number {
        font-size: 1.55rem;
        font-weight: 800;
        color: #c2185b;
        margin-top: 2px;
    }
    .transparency-note {
        margin-top: 12px;
        font-size: 0.80rem;
        color: #7b5865;
        font-style: italic;
        line-height: 1.4;
        background: rgba(255, 255, 255, 0.65);
        padding: 8px 10px;
        border-radius: 8px;
    }

    /* Rosa gepunkteter Rahmen um den Graphen (linke Spalte) */
    div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] {
        border: 3px dotted #f06292;
        border-radius: 18px;
        padding: 16px;
        background-color: #ffffff;
        box-shadow: 0 6px 20px rgba(240, 98, 146, 0.12);
    }

    /* Schieberegler-Griff als Blutstropfen (Streamlit / BaseWeb Slider) */
    div[data-baseweb="slider"] div[role="slider"] {
        width: 20px !important;
        height: 20px !important;
        background: linear-gradient(135deg, #e91e63 0%, #ad1457 100%) !important;
        border-radius: 0 50% 50% 50% !important;
        transform: rotate(45deg) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 3px 8px rgba(173, 20, 87, 0.45) !important;
    }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] {
        color: #ad1457 !important;
    }
    /* Aktive Slider-Spur in Rosa */
    div[data-baseweb="slider"] > div > div > div:nth-child(2) {
        background-color: #ec407a !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #880e4f !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #d81b60 !important;
        color: #d81b60 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 1. EMPIRISCHE OGTT-MESSWERTE (GLUKOSE & INSULIN)
# =====================================================================
t_eval = np.array([0, 10, 20, 30, 60, 90, 120])

# Glukose-Messwerte [mg/dL]
empirical_glucose = {
    'Follikelphase': np.array([81.6, 125.0, 140.0, 135.0, 105.0, 92.0, 85.0]),
    'Lutealphase':   np.array([87.1, 135.0, 150.0, 148.0, 122.0, 105.0, 94.0]),
    'Kontrazeptiva': np.array([85.2, 141.2, 161.5, 153.8, 131.0, 110.5, 98.4])
}

# Insulin-Messwerte [uU/mL]
empirical_insulin = {
    'Follikelphase': np.array([7.5, 32.0, 48.0, 42.0, 22.0, 14.0, 9.0]),
    'Lutealphase':   np.array([9.2, 44.0, 60.0, 56.0, 38.0, 24.0, 13.0]),
    'Kontrazeptiva': np.array([10.5, 41.6, 62.4, 54.6, 28.6, 18.2, 11.7])
}

# =====================================================================
# 2. GEFITTETE PARAMETER & INTEGRALE MODELLEINSTELLUNGEN
# =====================================================================
A = 1.8
k = 0.06

fit_parameters = {
    'Follikelphase': {
        'Gb': 81.6, 'Ib': 7.5,
        'p1': 0.15936, 'p2': 0.10000, 'p3': 0.000019,
        'S_G': 0.15936, 'S_I_scaled': 1.915
    },
    'Lutealphase': {
        'Gb': 87.1, 'Ib': 9.2,
        'p1': 0.14410, 'p2': 0.10000, 'p3': 0.000001,
        'S_G': 0.14410, 'S_I_scaled': 0.100
    },
    'Kontrazeptiva': {
        'Gb': 85.2, 'Ib': 10.5,
        'p1': 0.11685, 'p2': 0.09032, 'p3': 0.000001,
        'S_G': 0.11685, 'S_I_scaled': 0.111
    }
}

# =====================================================================
# 3. DAS MATHEMATISCHE ODE-SYSTEM (BERGMAN-MINIMAL-MODELL)
# =====================================================================
def bergman_minimal_model(t, y, params, Gb, Ib, G_meal_rate, insulin_input_func):
    G, X = y
    p1, p2, p3 = params
    I_t = insulin_input_func(t)
    J_t = G_meal_rate(t)
    dG_dt = -p1 * (G - Gb) - X * G + J_t
    dX_dt = -p2 * X + p3 * max(0, I_t - Ib)
    return [dG_dt, dX_dt]

def meal_absorption_rate(t, A=A, k=k):
    return A * t * np.exp(-k * t)

def simulate_ogtt_curve(p1, p2, p3, Gb, Ib, insulin_points, t_span=(0, 120), n_points=241):
    t_sim = np.linspace(t_span[0], t_span[1], n_points)
    insulin_func = interp1d(t_eval, insulin_points, kind='linear', fill_value='extrapolate')
    sol = solve_ivp(
        bergman_minimal_model,
        t_span,
        [Gb, 0.0],
        args=([p1, p2, p3], Gb, Ib, meal_absorption_rate, insulin_func),
        t_eval=t_sim,
        method='RK45'
    )
    return sol.t, sol.y[0]

# =====================================================================
# 4. HORMON- & GEWICHTUNGS-INTERPOLATIONEN FUER DEN ZYKLUSTAG (1-28)
# =====================================================================
days_cycle = np.arange(1, 29)
estrogen_curve = 50 + 150 * np.exp(-((days_cycle - 12)/3.5)**2) + 80 * np.exp(-((days_cycle - 21)/4)**2)
progesterone_curve = 0.5 + 1.2 * np.exp(-((days_cycle - 12)/4)**2) + 12.0 * np.exp(-((days_cycle - 21)/4)**2)

get_estrogen_at_day = interp1d(days_cycle, estrogen_curve, kind='linear', fill_value="extrapolate")
get_progesterone_at_day = interp1d(days_cycle, progesterone_curve, kind='linear', fill_value="extrapolate")

def get_cycle_weights(day):
    day = np.clip(day, 1, 28)
    if day <= 14:
        weight_luteal = 0.0
    else:
        weight_luteal = (day - 14) / 14.0
    weight_follikel = 1.0 - weight_luteal
    return weight_follikel, weight_luteal

# =====================================================================
# 5. KOPFBEREICH DER WEB-APP
# =====================================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🩸 In-silico Glukose-Insulin-Dynamik (OGTT)</div>
    <p class="hero-subtitle">
        Interaktive Simulation des erweiterten Bergman-Minimal-Modells im Verlauf des Menstruationszyklus
        und unter kombinierten oralen Kontrazeptiva (KOK) auf Basis klinischer Studiendaten.
    </p>
</div>
""", unsafe_allow_html=True)

tab_cycle, tab_study = st.tabs([
    "🩸 Zyklustag-Simulator & Hormonprofil",
    "📊 Klinischer Studienvergleich (Datengetreu)"
])

# =====================================================================
# TAB 1: ZYKLUSTAG-SIMULATOR & LIVE-KENNZAHLEN-BOX
# =====================================================================
with tab_cycle:
    col_graph, col_controls = st.columns([1.85, 1.0], gap="large")

    with col_controls:
        st.markdown('<div class="control-card"><div class="card-heading">🎛️ Physiologische Steuerung</div>', unsafe_allow_html=True)
        regime = st.selectbox(
            "Hormonelles Regime wählen:",
            ["Natürlicher Menstruationszyklus", "Orale Kontrazeptiva (KOK)"],
            key="regime_select"
        )

        is_kok = (regime == "Orale Kontrazeptiva (KOK)")
        slider_label = "Zyklustag (Inaktiv unter KOK)" if is_kok else "Zyklustag (Tag 1 – 28):"
        selected_day = st.slider(
            slider_label,
            min_value=1,
            max_value=28,
            value=8,
            disabled=is_kok,
            key="cycle_day_slider"
        )
        if is_kok:
            st.caption("ℹ️ Unter KOK findet eine konstante exogene Hormonzufuhr statt (keine zyklischen Tagesschwankungen).")
        st.markdown('</div>', unsafe_allow_html=True)

        # Berechnung der Parameter je nach Auswahl
        if is_kok:
            p = fit_parameters['Kontrazeptiva']
            p1, p2, p3 = p['p1'], p['p2'], p['p3']
            Gb, Ib = p['Gb'], p['Ib']
            si_val = p['S_I_scaled']
            ins_points = empirical_insulin['Kontrazeptiva']
            phase_name = "Kombinierte orale Kontrazeptiva"
            hormone_dom = "Exogen: Ethinylestradiol + Levonorgestrel"
            e2_text = "Konstant (exogen supprimiert)"
            p4_text = "Konstant (exogen supprimiert)"
            curve_color = "#2e7d32"
            ref_points = empirical_glucose['Kontrazeptiva']
            ref_label = "Empirische OGTT-Messwerte (KOK, Godsland et al.)"
            note_text = "Direkt an klinischen OGTT-Daten unter KOK (150 µg LNG / 30 µg EE) gefittete Parameter."
        else:
            w_foll, w_lut = get_cycle_weights(selected_day)
            pf = fit_parameters['Follikelphase']
            pl = fit_parameters['Lutealphase']

            p1 = w_foll * pf['p1'] + w_lut * pl['p1']
            p2 = w_foll * pf['p2'] + w_lut * pl['p2']
            p3 = w_foll * pf['p3'] + w_lut * pl['p3']
            Gb = w_foll * pf['Gb'] + w_lut * pl['Gb']
            Ib = w_foll * pf['Ib'] + w_lut * pl['Ib']
            si_val = w_foll * pf['S_I_scaled'] + w_lut * pl['S_I_scaled']
            ins_points = w_foll * empirical_insulin['Follikelphase'] + w_lut * empirical_insulin['Lutealphase']

            e2_val = float(get_estrogen_at_day(selected_day))
            p4_val = float(get_progesterone_at_day(selected_day))
            e2_text = f"ca. {e2_val:.0f} pg/mL"
            p4_text = f"ca. {p4_val:.1f} ng/mL"

            if selected_day <= 12:
                phase_name = f"Follikelphase (Tag {selected_day})"
                hormone_dom = "Östrogen-dominant (hohe Insulinsensitivität)"
                curve_color = "#1565c0"
            elif selected_day <= 15:
                phase_name = f"Ovulatorischer Übergang (Tag {selected_day})"
                hormone_dom = "Östradiol-Peak & beginnender Progesteron-Anstieg"
                curve_color = "#8e24aa"
            else:
                phase_name = f"Lutealphase (Tag {selected_day})"
                hormone_dom = "Progesteron-dominant (physiologisch reduzierte S_I)"
                curve_color = "#c62828"

            if selected_day <= 14:
                ref_points = empirical_glucose['Follikelphase']
                ref_label = "Klinische Referenz: Follikelphase (Bennal & Kerure)"
            else:
                ref_points = empirical_glucose['Lutealphase']
                ref_label = "Klinische Referenz: Lutealphase (Bennal & Kerure)"

            note_text = "Modellierte Interpolation zwischen den klinisch gemessenen Zeitpunkten Tag 8 (Follikelphase) und Tag 22 (Lutealphase)."

        # Hellrosa Messwerte-Box
        st.markdown(f"""
        <div class="metrics-card">
            <div class="card-heading">📋 Physiologische Kennzahlen</div>
            <div class="si-highlight">
                <div class="si-highlight-label">Erwartete Insulinsensitivität (S_I)</div>
                <div class="si-highlight-number">{si_val:.3f} <span style="font-size:0.85rem; font-weight:600;">× 10⁻⁴ min⁻¹/(µU/mL)</span></div>
            </div>
            <div class="metric-row">
                <span class="metric-label">Aktuelle Phase</span>
                <span class="metric-value">{phase_name}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Dominante Hormonlage</span>
                <span class="metric-value" style="font-size:0.90rem; text-align:right; max-width:58%;">{hormone_dom}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Östradiol (E₂)</span>
                <span class="metric-value">{e2_text}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Progesteron (P₄)</span>
                <span class="metric-value">{p4_text}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Nüchternglukose (G_b)</span>
                <span class="metric-value">{Gb:.1f} mg/dL</span>
            </div>
            <div class="transparency-note">
                ℹ️ {note_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_graph:
        t_sim, G_sim = simulate_ogtt_curve(p1, p2, p3, Gb, Ib, ins_points)

        fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=120)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fffcfd')

        ax.plot(
            t_sim, G_sim,
            color=curve_color, linewidth=3.0,
            label=f"Simulierter OGTT-Verlauf ({phase_name})"
        )
        ax.scatter(
            t_eval, ref_points,
            color=curve_color, edgecolor='#ffffff', linewidth=1.3,
            s=85, zorder=5, label=ref_label
        )

        ax.set_title(
            f"Oraler Glukosetoleranztest (75 g OGTT) – {phase_name}",
            fontsize=13, fontweight='bold', color='#880e4f', pad=14
        )
        ax.set_xlabel("Zeit nach oraler Glukoseaufnahme [Minuten]", fontsize=11, color='#4a2533', labelpad=8)
        ax.set_ylabel("Plasmaglukose [mg/dL]", fontsize=11, color='#4a2533', labelpad=8)
        ax.set_xlim(-3, 123)
        ax.set_ylim(65, 185)
        ax.set_xticks([0, 10, 20, 30, 60, 90, 120])
        ax.grid(True, linestyle='--', alpha=0.45, color='#f48fb1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#f06292')
        ax.spines['bottom'].set_color('#f06292')
        ax.legend(frameon=True, facecolor='#fff0f5', edgecolor='#f8bbd0', fontsize=9.5, loc='upper right')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =====================================================================
# TAB 2: REIN DATENGETREUER KLINISCHER STUDIENVERGLEICH
# =====================================================================
with tab_study:
    col_study_plot, col_study_ctrl = st.columns([1.85, 1.0], gap="large")

    with col_study_ctrl:
        st.markdown('<div class="control-card"><div class="card-heading">🔬 Studienkohorten & Streuung</div>', unsafe_allow_html=True)
        selected_group = st.radio(
            "Klinische Vergleichsgruppe:",
            ["Alle drei im Vergleich", "Follikelphase", "Lutealphase", "Kontrazeptiva"],
            key="study_group_radio"
        )
        show_ci = st.checkbox("95-%-Monte-Carlo-Konfidenzband anzeigen", value=True, key="show_ci_check")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="metrics-card">
            <div class="card-heading">📚 Empirische Datenbasis</div>
            <div class="metric-row">
                <span class="metric-label">Follikelphase (S_I)</span>
                <span class="metric-value" style="color:#1565c0;">1,915 × 10⁻⁴</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Lutealphase (S_I)</span>
                <span class="metric-value" style="color:#c62828;">0,100 × 10⁻⁴</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Kontrazeptiva / KOK (S_I)</span>
                <span class="metric-value" style="color:#2e7d32;">0,111 × 10⁻⁴</span>
            </div>
            <div class="transparency-note">
                Quellen: Bennal & Kerure (2013) für Follikel- und Lutealphase; Godsland et al. (1990) für kombinierte orale Kontrazeptiva.
                Externe IVGTT-Validierung: Escalante Pulido & Alpizar Salazar (1999).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_study_plot:
        fig2, ax2 = plt.subplots(figsize=(9.5, 5.6), dpi=120)
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#fffcfd')

        groups_to_plot = (
            ['Follikelphase', 'Lutealphase', 'Kontrazeptiva']
            if selected_group == "Alle drei im Vergleich"
            else [selected_group]
        )
        colors_map = {
            'Follikelphase': '#1565c0',
            'Lutealphase': '#c62828',
            'Kontrazeptiva': '#2e7d32'
        }

        rng = np.random.default_rng(42)
        for grp in groups_to_plot:
            p = fit_parameters[grp]
            c = colors_map[grp]
            t_s, g_s = simulate_ogtt_curve(
                p['p1'], p['p2'], p['p3'], p['Gb'], p['Ib'], empirical_insulin[grp]
            )
            ax2.plot(t_s, g_s, color=c, linewidth=2.8, label=f"{grp} (Modell-Fit, S_I={p['S_I_scaled']:.2f})")
            ax2.scatter(t_eval, empirical_glucose[grp], color=c, edgecolor='#ffffff', linewidth=1.2, s=75, zorder=5)

            if show_ci:
                mc_curves = []
                for _ in range(80):
                    p1_mc = max(0.01, rng.normal(p['p1'], p['p1'] * 0.12))
                    p3_mc = max(1e-7, rng.normal(p['p3'], p['p3'] * 0.20))
                    _, g_mc = simulate_ogtt_curve(
                        p1_mc, p['p2'], p3_mc, p['Gb'], p['Ib'], empirical_insulin[grp]
                    )
                    mc_curves.append(g_mc)
                mc_arr = np.array(mc_curves)
                low_b = np.percentile(mc_arr, 2.5, axis=0)
                high_b = np.percentile(mc_arr, 97.5, axis=0)
                ax2.fill_between(t_s, low_b, high_b, color=c, alpha=0.14)

        ax2.set_title("Empirischer Gruppenvergleich & Modell-Fit (75 g OGTT)", fontsize=13, fontweight='bold', color='#880e4f', pad=14)
        ax2.set_xlabel("Zeit [Minuten]", fontsize=11, color='#4a2533', labelpad=8)
        ax2.set_ylabel("Plasmaglukose [mg/dL]", fontsize=11, color='#4a2533', labelpad=8)
        ax2.set_xlim(-3, 123)
        ax2.set_ylim(65, 195)
        ax2.set_xticks([0, 10, 20, 30, 60, 90, 120])
        ax2.grid(True, linestyle='--', alpha=0.45, color='#f48fb1')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#f06292')
        ax2.spines['bottom'].set_color('#f06292')
        ax2.legend(frameon=True, facecolor='#fff0f5', edgecolor='#f8bbd0', fontsize=9.5, loc='upper right')

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
