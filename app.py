import streamlit as st
import numpy as np
import pandas as pd
from engine import (
    devig_mult, devig_power, devig_shin, hk_overround,
    dc_matrix, nb_matrix, build_payoff_fn, eval_bet,
    fit_scen_a, baker_mchale_kelly, same_match_corr,
    clv_pct, proper_scores
)

st.set_page_config(page_title="Quant Football Value Engine", layout="wide")
st.title("Quant Football Value & Risk Engine")
st.caption("Kerangka Analisis Kuantitatif: Power/Shin De-Vigging, Dixon-Coles, Negative Binomial, & Baker-McHale Shrunken Kelly")

# Sidebar: Parameter Global Model
with st.sidebar:
    st.header("Parameter Global Model")
    rho_dc = st.number_input("Korelasi Dixon-Coles (rho)", value=-0.08, step=0.01, format="%.2f")
    alpha_nb = st.number_input("Overdispersi NegBin (alpha)", value=0.12, min_value=0.0, max_value=0.5, step=0.02, format="%.2f")
    kelly_frac = st.slider("Fraksi Kelly Konservatif", min_value=0.10, max_value=0.50, value=0.25, step=0.05)
    stake_cap = st.slider("Batas Atas Stake (% Bankroll)", min_value=0.5, max_value=5.0, value=2.5, step=0.25) / 100.0
    bankroll = st.number_input("Total Bankroll (Unit / Mata Uang)", value=10000.0, step=500.0)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Kalibrasi Skenario A & De-Vigging",
    "2. Evaluasi Value & Kelly Sizing",
    "3. Audit Korelasi Same-Match",
    "4. Audit CLV & Proper Scoring (RPS)"
])

# ==========================================
# TAB 1: KALIBRASI SKENARIO A & DE-VIGGING
# ==========================================
with tab1:
    st.subheader("Ekstraksi Probabilitas Wajar 1X2 & Kalibrasi Laju Gol Skenario A")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Odds 1X2 Eksternal (Desimal)**")
        o_home = st.number_input("Odds Home (1)", value=2.15, min_value=1.01, step=0.01)
        o_draw = st.number_input("Odds Draw (X)", value=3.60, min_value=1.01, step=0.01)
        o_away = st.number_input("Odds Away (2)", value=3.25, min_value=1.01, step=0.01)

    with col2:
        st.markdown("**Input Garis Total Gol Papan HK (Untuk Kalibrasi Skenario A)**")
        ou_cal_line = st.number_input("Garis O/U Utama", value=2.75, step=0.25)
        hk_cal_over = st.number_input("Odds HK Over", value=0.87, min_value=0.05, step=0.01)
        hk_cal_under = st.number_input("Odds HK Under", value=1.07, min_value=0.05, step=0.01)

    p_m, vig_1x2 = devig_mult([o_home, o_draw, o_away])
    p_p, k_pow = devig_power([o_home, o_draw, o_away])
    p_s, z_shin = devig_shin([o_home, o_draw, o_away])
    vig_ou = hk_overround(hk_cal_over, hk_cal_under)

    df_devig = pd.DataFrame({
        "Metode": ["Multiplikatif", f"Power (k={k_pow:.3f})", f"Shin (z={z_shin:.4f})"],
        "Home (%)": [round(p_m[0]*100, 2), round(p_p[0]*100, 2), round(p_s[0]*100, 2)],
        "Draw (%)": [round(p_m[1]*100, 2), round(p_p[1]*100, 2), round(p_s[1]*100, 2)],
        "Away (%)": [round(p_m[2]*100, 2), round(p_p[2]*100, 2), round(p_s[2]*100, 2)],
    })

    st.info(f"Overround 1X2 Eksternal: **{vig_1x2*100:.2f}%** | Overround O/U Papan HK: **{vig_ou*100:.2f}%**")
    st.dataframe(df_devig, use_container_width=True, hide_index=True)

    if st.button("Hitung Laju Gol Kalibrasi Skenario A (Power Method)"):
        lam_a_fit, mu_a_fit = fit_scen_a(p_p, ou_cal_line, hk_cal_over, hk_cal_under, rho=rho_dc)
        st.session_state["lam_a"] = round(lam_a_fit, 2)
        st.session_state["mu_a"] = round(mu_a_fit, 2)
        st.success(f"Hasil Kalibrasi Skenario A: **lambda_A = {lam_a_fit:.2f}** | **mu_A = {mu_a_fit:.2f}** (Total = {lam_a_fit+mu_a_fit:.2f} gol)")

# ==========================================
# TAB 2: EVALUASI VALUE & BAKER-MCHALE KELLY
# ==========================================
with tab2:
    st.subheader("Simulasi Expected Value (Skenario A vs B) & Sizing")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Skenario A (Basis Pasar)**")
        lam_a = st.number_input("Lambda A (Home)", value=st.session_state.get("lam_a", 1.55), min_value=0.1, step=0.05)
        mu_a = st.number_input("Mu A (Away)", value=st.session_state.get("mu_a", 1.20), min_value=0.1, step=0.05)

    with c2:
        st.markdown("**Skenario B (xG & Konteks Skuad)**")
        lam_b = st.number_input("Lambda B (Home)", value=1.32, min_value=0.1, step=0.05)
        mu_b = st.number_input("Mu B (Away)", value=1.12, min_value=0.1, step=0.05)

    with c3:
        st.markdown("**Pilihan Pasar Papan HK**")
        mkt_type = st.selectbox("Tipe Pasar", ["Under", "Over", "Home HDP", "Away HDP"])
        bet_line = st.number_input("Garis (Line)", value=2.75, step=0.25)
        bet_hk = st.number_input("Odds HK Pilihan", value=1.07, min_value=0.05, step=0.01)
        opp_hk = st.number_input("Odds HK Lawan (Cek Vig)", value=0.87, min_value=0.05, step=0.01)

    vig_mkt = hk_overround(bet_hk, opp_hk)
    Ma = dc_matrix(lam_a, mu_a, rho=rho_dc)
    Mb = dc_matrix(lam_b, mu_b, rho=rho_dc)
    Mnb = nb_matrix(lam_b, mu_b, alpha=alpha_nb, rho=rho_dc)

    fn = build_payoff_fn(mkt_type, bet_line, bet_hk)
    eva, sda, _, _ = eval_bet(Ma, fn)
    evb, sdb, sr_b, nl_b = eval_bet(Mb, fn)
    evnb, _, _, _ = eval_bet(Mnb, fn)
    kelly = baker_mchale_kelly(eva, evb, sdb, frac=kelly_frac, cap=stake_cap)

    # Warning Gating Overround
    if vig_mkt > 0.035:
        st.warning(f"Peringatan Overround Tinggi ({vig_mkt*100:.2f}% > 3,50%): Waspadai penggerusan EV pada papan margin lebar.")
    else:
        st.info(f"Overround Pasar Terpilih: **{vig_mkt*100:.2f}%** (Efisien)")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EV Skenario A", f"{eva*100:+.2f}%")
    m2.metric("EV Skenario B", f"{evb*100:+.2f}%")
    m3.metric(f"EV NegBin (a={alpha_nb})", f"{evnb*100:+.2f}%")
    m4.metric("Non-Loss (Scen B)", f"{nl_b*100:.1f}%")
    m5.metric("BM Shrinkage", f"{kelly['bm_alpha']:.2f}")

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("EV / SD (Scen B)", f"{sr_b:+.3f}")
    k2.metric("Full Kelly (Tengah)", f"{kelly['full_kelly']*100:.2f}%")
    k3.metric("Rekomendasi Stake (%)", f"{kelly['stake']*100:.2f}%")
    k4.metric("Nominal Taruhan", f"{kelly['stake']*bankroll:,.2f}")

    # Matriks Probabilitas 1X2 Skenario B
    ph_b, pd_b, pa_b = np.sum(np.tril(Mb, -1)), np.sum(np.diag(Mb)), np.sum(np.triu(Mb, 1))
    st.caption(f"Implikasi 1X2 Skenario B: Home {ph_b*100:.1f}% | Draw {pd_b*100:.1f}% | Away {pa_b*100:.1f}%")

# ==========================================
# TAB 3: AUDIT KORELASI SAME-MATCH
# ==========================================
with tab3:
    st.subheader("Deteksi Korelasi Imbal Hasil Antar-Pasar dalam Satu Pertandingan")
    col_a, col_b = st.columns(2)
    with col_a:
        lam_corr = st.number_input("Lambda Skenario B", value=2.55, min_value=0.1, step=0.05, key="corr_l")
        t1 = st.selectbox("Pasar 1", ["Home HDP", "Away HDP", "Over", "Under"], index=1)
        l1 = st.number_input("Line 1", value=1.75, step=0.25, key="l1")
        o1 = st.number_input("Odds HK 1", value=0.97, min_value=0.05, step=0.01, key="o1")
    with col_b:
        mu_corr = st.number_input("Mu Skenario B", value=1.05, min_value=0.1, step=0.05, key="corr_m")
        t2 = st.selectbox("Pasar 2", ["Under", "Over", "Home HDP", "Away HDP"], index=0)
        l2 = st.number_input("Line 2", value=3.50, step=0.25, key="l2")
        o2 = st.number_input("Odds HK 2", value=1.08, min_value=0.05, step=0.01, key="o2")

    M_corr = dc_matrix(lam_corr, mu_corr, rho=rho_dc)
    fn1 = build_payoff_fn(t1, l1, o1)
    fn2 = build_payoff_fn(t2, l2, o2)
    ev1, sd1, _, _ = eval_bet(M_corr, fn1)
    ev2, sd2, _, _ = eval_bet(M_corr, fn2)
    corr_val = same_match_corr(M_corr, fn1, fn2)

    st.metric("Korelasi Imbal Hasil (r)", f"{corr_val:+.3f}")
    st.write(f"**EV Pasar 1:** {ev1*100:+.2f}% (SD {sd1:.3f}) | **EV Pasar 2:** {ev2*100:+.2f}% (SD {sd2:.3f})")
    if corr_val >= 0.25:
        st.error("Korelasi positif tinggi: Risiko kalah ganda sangat besar. Ambil hanya satu posisi dengan rasio EV/SD tertinggi.")
    elif corr_val <= -0.25:
        st.warning("Korelasi negatif kuat: Kedua posisi cenderung saling menegasikan sebagian imbal hasil.")
    else:
        st.success("Korelasi rendah/moderat: Relatif aman dikombinasikan jika masing-masing memiliki alpha_BM tinggi.")

# ==========================================
# TAB 4: AUDIT CLV & PROPER SCORING RULES
# ==========================================
with tab4:
    st.subheader("Validasi Jangka Panjang: Closing Line Value (CLV) & Kalibrasi RPS")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("**Kalkulator CLV%**")
        taken_hk = st.number_input("Odds HK Saat Eksekusi", value=1.07, min_value=0.01, step=0.01)
        close_hk = st.number_input("Odds HK Penutup Wajar (Pinnacle No-Vig)", value=0.95, min_value=0.01, step=0.01)
        clv = clv_pct(taken_hk, close_hk)
        st.metric("CLV (%)", f"{clv*100:+.2f}%")

    with cb:
        st.markdown("**Kalkulator Proper Scoring Rules (1X2)**")
        p_h = st.number_input("Probabilitas Model Home (%)", value=40.0, min_value=0.1, max_value=99.8) / 100.0
        p_d = st.number_input("Probabilitas Model Draw (%)", value=28.0, min_value=0.1, max_value=99.8) / 100.0
        p_a = st.number_input("Probabilitas Model Away (%)", value=32.0, min_value=0.1, max_value=99.8) / 100.0
        actual = st.selectbox("Hasil Aktual", ["Home Win", "Draw", "Away Win"], index=1)
        idx = ["Home Win", "Draw", "Away Win"].index(actual)
        sc = proper_scores([p_h, p_d, p_a], idx)
        st.write(f"**RPS:** {sc['rps']:.4f} | **Brier Score:** {sc['brier']:.4f} | **Log-Loss:** {sc['log_loss']:.4f}")