import numpy as np
from scipy.stats import poisson, nbinom
from scipy.optimize import brentq, minimize


# ==========================================
# 1. DE-VIGGING & OVERROUND PAPAN PASARAN
# ==========================================

def devig_mult(odds):
    """Normalisasi multiplikatif standar beserta nilai overround."""
    inv = np.array([1.0 / o for o in odds], dtype=float)
    return inv / inv.sum(), float(inv.sum() - 1.0)


def devig_power(odds):
    """Power Method (Clarke, Kovalchik, Ingram) untuk mengatasi bias favorit-longshot."""
    inv = np.array([1.0 / o for o in odds], dtype=float)
    if inv.sum() <= 1.0:
        return inv / inv.sum(), 1.0
    k = brentq(lambda x: np.sum(inv ** x) - 1.0, 0.5, 5.0)
    return inv ** k, float(k)


def devig_shin(odds):
    """Metode Shin berbasis estimasi proporsi informasi insider (z)."""
    inv = np.array([1.0 / o for o in odds], dtype=float)
    pi_sum = inv.sum()
    if pi_sum <= 1.0:
        return inv / pi_sum, 0.0

    def f(z):
        p = (np.sqrt(z**2 + 4 * (1 - z) * (inv**2) / pi_sum) - z) / (2 * (1 - z))
        return p.sum() - 1.0

    z = brentq(f, 0.0, 0.5)
    p = (np.sqrt(z**2 + 4 * (1 - z) * (inv**2) / pi_sum) - z) / (2 * (1 - z))
    return p, float(z)


def hk_overround(hk_side1, hk_side2):
    """Menghitung overround papan dua arah berformat odds Hong Kong (desimal = 1 + HK)."""
    return float((1.0 / (1.0 + hk_side1) + 1.0 / (1.0 + hk_side2) - 1.0))


# ==========================================
# 2. MATRIKS PROBABILITAS SKOR (DC & NEGBIN)
# ==========================================

def dc_matrix(lam, mu, rho=-0.08, max_g=14):
    """Matriks Bivariate Poisson dengan koreksi skor rendah Dixon-Coles."""
    M = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson.pmf(i, lam) * poisson.pmf(j, mu)
            if i == 0 and j == 0:
                tau = 1.0 - lam * mu * rho
            elif i == 0 and j == 1:
                tau = 1.0 + lam * rho
            elif i == 1 and j == 0:
                tau = 1.0 + mu * rho
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            M[i, j] = max(0.0, p * tau)
    return M / M.sum()


def nb_matrix(lam, mu, alpha=0.12, rho=-0.08, max_g=14):
    """Matriks Negative Binomial + koreksi Dixon-Coles untuk menguji overdispersi/tail risk."""
    if alpha <= 0:
        return dc_matrix(lam, mu, rho=rho, max_g=max_g)
    r = 1.0 / alpha
    p_h = r / (r + lam)
    p_a = r / (r + mu)
    M = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = nbinom.pmf(i, r, p_h) * nbinom.pmf(j, r, p_a)
            if i == 0 and j == 0:
                tau = 1.0 - lam * mu * rho
            elif i == 0 and j == 1:
                tau = 1.0 + lam * rho
            elif i == 1 and j == 0:
                tau = 1.0 + mu * rho
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            M[i, j] = max(0.0, p * tau)
    return M / M.sum()


# ==========================================
# 3. FUNGSI PAYOFF GARIS SEPEREMPAT & PENUH
# ==========================================

def ah_payoff(diff, line, hk_odds):
    """
    Payoff Asian Handicap dalam unit taruhan.
    diff: (skor tim pilihan - skor lawan).
    line: garis handicap tim pilihan (misal -1.25 atau +0.5).
    """
    if abs((line * 4) % 2) == 1:
        return 0.5 * ah_payoff(diff, line - 0.25, hk_odds) + 0.5 * ah_payoff(diff, line + 0.25, hk_odds)
    net = diff + line
    if net > 0:
        return hk_odds
    elif net == 0:
        return 0.0
    else:
        return -1.0


def ou_payoff(total, line, hk_odds, is_under=True):
    """Payoff Over/Under (mendukung garis bola penuh, setengah, dan seperempat)."""
    if abs((line * 4) % 2) == 1:
        return 0.5 * ou_payoff(total, line - 0.25, hk_odds, is_under) + 0.5 * ou_payoff(total, line + 0.25, hk_odds, is_under)
    if is_under:
        if total < line:
            return hk_odds
        elif total == line:
            return 0.0
        else:
            return -1.0
    else:
        if total > line:
            return hk_odds
        elif total == line:
            return 0.0
        else:
            return -1.0


def build_payoff_fn(market_type, line, hk_odds):
    """Helper pembuat fungsi payoff untuk pilihan Home, Away, Over, atau Under."""
    if market_type == "Home HDP":
        return lambda i, j: ah_payoff(i - j, line, hk_odds)
    elif market_type == "Away HDP":
        return lambda i, j: ah_payoff(j - i, line, hk_odds)
    elif market_type == "Over":
        return lambda i, j: ou_payoff(i + j, line, hk_odds, is_under=False)
    else:
        return lambda i, j: ou_payoff(i + j, line, hk_odds, is_under=True)


# ==========================================
# 4. EVALUASI EV, KALIBRASI & KELLY SHRINKAGE
# ==========================================

def eval_bet(M, fn):
    """Menghitung EV, Simpangan Baku (SD), Rasio EV/SD, dan Probabilitas Non-Loss."""
    n = M.shape[0]
    vals = np.array([fn(i, j) for i in range(n) for j in range(n)])
    probs = np.array([M[i, j] for i in range(n) for j in range(n)])
    ev = float(np.sum(vals * probs))
    sd = float(np.sqrt(np.sum(((vals - ev) ** 2) * probs)))
    p_nonloss = float(np.sum(probs[vals >= 0]))
    return ev, sd, (ev / sd if sd > 0 else 0.0), p_nonloss


def fit_scen_a(p_1x2, ou_line, ou_over_hk, ou_under_hk, rho=-0.08):
    """Kalibrasi laju gol Skenario A (lam_A, mu_A) agar cocok dengan probabilitas wajar 1X2 & O/U."""
    p_ou_fair, _ = devig_power([1.0 + ou_over_hk, 1.0 + ou_under_hk])
    p_over_fair = p_ou_fair[0]

    def loss(x):
        lam, mu = x
        if lam <= 0.05 or mu <= 0.05:
            return 1e6
        M = dc_matrix(lam, mu, rho=rho, max_g=12)
        ph = np.sum(np.tril(M, -1))
        pd_prob = np.sum(np.diag(M))
        pa = np.sum(np.triu(M, 1))
        # Probabilitas Over ekivalen termasuk garis seperempat (payoff unit 1.0 -> skala [0, 1])
        ev_unit, _, _, _ = eval_bet(M, lambda i, j: ou_payoff(i + j, ou_line, 1.0, is_under=False))
        p_o_prob = (ev_unit + 1.0) / 2.0
        return ((ph - p_1x2[0])**2 + (pd_prob - p_1x2[1])**2 + (pa - p_1x2[2])**2) * 10.0 + (p_o_prob - p_over_fair)**2

    res = minimize(loss, [1.6, 1.2], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def baker_mchale_kelly(eva, evb, sdb, frac=0.25, cap=0.025):
    """
    Menghitung ukuran taruhan Baker-McHale Shrunken Kelly dari selisih Skenario A dan B,
    lalu menerapkan pengali fraksional konservatif serta batas atas (cap) bankroll.
    """
    ev_mid = 0.5 * (eva + evb)
    if ev_mid <= 0:
        return {"ev_mid": ev_mid, "bm_alpha": 0.0, "full_kelly": 0.0, "shrunk_kelly": 0.0, "stake": 0.0}
    sig = abs(evb - eva) / 2.0
    f_mid = ev_mid / (sdb**2 + ev_mid**2)
    bm = (ev_mid**2) / (ev_mid**2 + sig**2) if (ev_mid**2 + sig**2) > 0 else 0.0
    f_shrunk = f_mid * bm
    stake = min(cap, frac * f_shrunk)
    return {
        "ev_mid": float(ev_mid),
        "bm_alpha": float(bm),
        "full_kelly": float(f_mid),
        "shrunk_kelly": float(f_shrunk),
        "stake": float(stake),
    }


def same_match_corr(M, fn1, fn2):
    """Menghitung korelasi imbal hasil antara dua pasar pada pertandingan yang sama."""
    n = M.shape[0]
    r1 = np.array([fn1(i, j) for i in range(n) for j in range(n)])
    r2 = np.array([fn2(i, j) for i in range(n) for j in range(n)])
    p = np.array([M[i, j] for i in range(n) for j in range(n)])
    m1, m2 = np.sum(p * r1), np.sum(p * r2)
    sd1 = np.sqrt(np.sum(p * (r1 - m1)**2))
    sd2 = np.sqrt(np.sum(p * (r2 - m2)**2))
    if sd1 == 0 or sd2 == 0:
        return 0.0
    return float(np.sum(p * (r1 - m1) * (r2 - m2)) / (sd1 * sd2))


# ==========================================
# 5. PROTOKOL AUDIT (CLV & PROPER SCORES)
# ==========================================

def clv_pct(taken_hk, closing_fair_hk):
    """Menghitung Closing Line Value (CLV%) terhadap harga penutup wajar tanpa vig."""
    return ((1.0 + taken_hk) / (1.0 + closing_fair_hk)) - 1.0


def proper_scores(probs_1x2, outcome_idx):
    """
    Menghitung Ranked Probability Score (RPS), Brier Score, dan Log-Loss.
    outcome_idx: 0 = Home Win, 1 = Draw, 2 = Away Win.
    """
    p = np.array(probs_1x2, dtype=float)
    p = p / p.sum()
    y = np.zeros(3)
    y[outcome_idx] = 1.0
    cdf_p = np.cumsum(p)
    cdf_y = np.cumsum(y)
    rps = 0.5 * np.sum((cdf_p[:-1] - cdf_y[:-1]) ** 2)
    brier = float(np.sum((p - y) ** 2))
    log_loss = float(-np.log(max(p[outcome_idx], 1e-12)))
    return {"rps": float(rps), "brier": brier, "log_loss": log_loss}