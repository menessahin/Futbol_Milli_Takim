import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Türkiye Milli Takım | Kadro Optimizasyonu",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0f1e; color: #e8eaf0; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.5px; }

.header-band {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5e 50%, #0d1b3e 100%);
    border-bottom: 2px solid #e63946;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex; align-items: center; gap: 1rem;
}
.header-title { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #fff; margin: 0; }
.header-sub { font-size: 0.8rem; color: #8899bb; letter-spacing: 1px; text-transform: uppercase; margin: 0; }

.pos-card {
    background: #111827; border: 1px solid #1e2d4a; border-radius: 8px;
    padding: 1rem; margin-bottom: 0.75rem; transition: border-color 0.2s;
}
.pos-card:hover { border-color: #e63946; }
.pos-card-title { font-family: 'Barlow Condensed', sans-serif; font-size: 1.1rem; font-weight: 600; color: #e63946; margin-bottom: 0.25rem; }
.player-name { font-size: 1.3rem; font-weight: 600; color: #ffffff; }
.player-meta { font-size: 0.75rem; color: #6b7fa3; margin-top: 0.2rem; }
.score-badge {
    background: #e63946; color: white;
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem; font-weight: 700;
    padding: 0.3rem 0.8rem; border-radius: 4px; float: right;
}

.stSlider > div > div > div { background: #e63946 !important; }
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #6b7fa3; }
.stTabs [aria-selected="true"] { color: #e63946 !important; }
.streamlit-expanderHeader { background: #111827 !important; color: #e8eaf0 !important; border-radius: 6px; }
.stDataFrame { background: #111827; }
[data-testid="stMetricValue"] { color: #e63946; font-family: 'Barlow Condensed', sans-serif; font-size: 2rem !important; }

.section-title {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem; font-weight: 700;
    color: #8899bb; text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid #e63946; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── VERİ ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("milli_takim_stats_full.xlsx")
    altay_idx = df[df['name'] == 'Altay Bayındır'].index
    if len(altay_idx) > 0:
        numeric_cols = df.select_dtypes(include='number').columns
        df.loc[altay_idx, numeric_cols] = None
    return df

df = load_data()

NEG_COLS = {'defending__dribbled_past', 'defending__fouls_committed', 'goalkeeping__error_led_to_goal'}

def minmax_normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def normalize_inverse(series):
    return 1 - minmax_normalize(series)

def local_normalize(df_raw, candidate_names, metric_cols):
    subset = df_raw[df_raw['name'].isin(candidate_names)].copy()
    norm = pd.DataFrame()
    norm['name'] = subset['name'].values

    for col in metric_cols:
        if col not in df_raw.columns:
            norm[col] = 0.5
            continue
        series = subset[col].fillna(subset[col].median() if subset[col].notna().any() else 0.5)
        if col in NEG_COLS:
            norm[col] = normalize_inverse(series).values
        else:
            norm[col] = minmax_normalize(series).values

    for meta in ['team', 'league', 'position_primary', 'age']:
        if meta in df_raw.columns:
            norm[meta] = subset[meta].values

    return norm.reset_index(drop=True)

def compute_score(player_row, metrics_weights):
    score = 0.0
    for col, (label, w) in metrics_weights.items():
        val = player_row.get(col, 0.5)
        if pd.isna(val):
            val = 0.5
        score += val * w
    return round(score * 100, 1)

def compute_all_scores(pos_name, pos_config, weights_override, df_raw):
    candidate_names = pos_config['candidates']
    metric_cols = list(pos_config['metrics'].keys())

    local_norm = local_normalize(df_raw, candidate_names, metric_cols)

    metrics = {}
    for col, (label, default_w) in pos_config['metrics'].items():
        metrics[col] = (label, weights_override.get(col, default_w))
    total_w = sum(w for _, w in metrics.values())
    if total_w > 0:
        metrics = {col: (lbl, w / total_w) for col, (lbl, w) in metrics.items()}

    scores = []
    for _, row in local_norm.iterrows():
        s = compute_score(row, metrics)
        scores.append({
            'name': row['name'],
            'score': s,
            'team': row.get('team', '—'),
            'league': row.get('league', '—'),
            'position_primary': row.get('position_primary', '—'),
        })

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores

# ── SCIPY'SİZ HUNGARIAN ALGORİTMASI ─────────────────────────────────────────
def _hungarian(cost_matrix):
    """
    Saf numpy ile Hungarian (Macar) algoritması.
    scipy.optimize.linear_sum_assignment ile aynı sonucu verir.
    Minimize eder — maximize için negatif matris ver.
    """
    C = cost_matrix.copy().astype(float)
    n, m = C.shape
    size = max(n, m)

    # Kare matrise pad et
    C_sq = np.full((size, size), np.max(C[C < 1e5]) * 2 if np.any(C < 1e5) else 1000.0)
    C_sq[:n, :m] = C

    # Adım 1: satır minimumlarını çıkar
    C_sq -= C_sq.min(axis=1, keepdims=True)
    # Adım 2: sütun minimumlarını çıkar
    C_sq -= C_sq.min(axis=0, keepdims=True)

    row_covered = np.zeros(size, dtype=bool)
    col_covered = np.zeros(size, dtype=bool)
    assignment = np.full(size, -1, dtype=int)  # assignment[row] = col

    def find_zeros():
        zeros = []
        for r in range(size):
            for c in range(size):
                if abs(C_sq[r, c]) < 1e-9:
                    zeros.append((r, c))
        return zeros

    for _ in range(size * size):
        # Atanmamış sıfırları bul ve ata
        row_assigned = np.zeros(size, dtype=bool)
        col_assigned = np.zeros(size, dtype=bool)
        assignment = np.full(size, -1, dtype=int)

        zeros = find_zeros()
        # Önce tek seçenekli satırları ata
        for r in range(size):
            row_zeros = [c for (rr, c) in zeros if rr == r]
            if len(row_zeros) == 1:
                c = row_zeros[0]
                if not col_assigned[c]:
                    assignment[r] = c
                    row_assigned[r] = True
                    col_assigned[c] = True

        # Kalan sıfırları ata
        for r, c in zeros:
            if not row_assigned[r] and not col_assigned[c]:
                assignment[r] = c
                row_assigned[r] = True
                col_assigned[c] = True

        assigned_count = np.sum(assignment >= 0)
        if assigned_count == size:
            break

        # Minimum satır sayısı ile tüm sıfırları örtecek çizgi seti bul
        # (Basitleştirilmiş: örtülmemiş minimum değeri güncelle)
        marked_rows = set()
        marked_cols = set()

        for r in range(size):
            if assignment[r] < 0:
                marked_rows.add(r)

        changed = True
        while changed:
            changed = False
            for r in marked_rows:
                for rr, c in zeros:
                    if rr == r and c not in marked_cols:
                        marked_cols.add(c)
                        changed = True
            for c in marked_cols:
                for r in range(size):
                    if assignment[r] == c and r not in marked_rows:
                        marked_rows.add(r)
                        changed = True

        covered_rows = set(range(size)) - marked_rows
        covered_cols = marked_cols

        uncovered_vals = [
            C_sq[r, c]
            for r in range(size) for c in range(size)
            if r not in covered_rows and c not in covered_cols
        ]
        if not uncovered_vals:
            break
        mn = min(uncovered_vals)

        for r in range(size):
            for c in range(size):
                if r not in covered_rows and c not in covered_cols:
                    C_sq[r, c] -= mn
                elif r in covered_rows and c in covered_cols:
                    C_sq[r, c] += mn

    # Orijinal boyuta kırp
    valid = [(r, assignment[r]) for r in range(size)
             if assignment[r] >= 0 and r < n and assignment[r] < m]
    if not valid:
        return [], []
    row_ind, col_ind = zip(*valid)
    return list(row_ind), list(col_ind)


# ── GLOBAL OPTİMİZASYON ─────────────────────────────────────────────────────
def optimize_lineup(positions, formation_order, weights_by_pos, df_raw):
    pos_names = formation_order
    player_pool = list(dict.fromkeys(
        p for pn in pos_names for p in positions[pn]['candidates']
    ))

    n_pos = len(pos_names)
    n_players = len(player_pool)

    pos_scores_matrix = np.full((n_pos, n_players), -1.0)
    all_scores = {}

    for i, pos_name in enumerate(pos_names):
        pos_config = positions[pos_name]
        weights_override = weights_by_pos.get(pos_name, {})
        candidate_scores = compute_all_scores(pos_name, pos_config, weights_override, df_raw)
        score_map = {d['name']: d for d in candidate_scores}
        all_scores[pos_name] = candidate_scores

        for j, player_name in enumerate(player_pool):
            if player_name in score_map:
                pos_scores_matrix[i][j] = score_map[player_name]['score']

    # Maximize için negatif, uygun olmayan için büyük pozitif ceza
    cost_matrix = np.where(pos_scores_matrix >= 0, -pos_scores_matrix, 1e6)
    row_ind, col_ind = _hungarian(cost_matrix)

    lineup = {}
    for i, j in zip(row_ind, col_ind):
        if pos_scores_matrix[i][j] >= 0:
            player_name = player_pool[j]
            pos_name = pos_names[i]
            score_detail = next(
                (d for d in all_scores[pos_name] if d['name'] == player_name),
                {'name': player_name, 'score': 0, 'team': '—', 'league': '—', 'position_primary': '—'}
            )
            lineup[pos_name] = score_detail

    return lineup, all_scores


# ── POZİSYONLAR ──────────────────────────────────────────────────────────────
POSITIONS = {
    "Kaleci": {
        "abbr": "KAL", "formation_slot": "gk", "eligible": ["Keeper"],
        "candidates": ["Altay Bayındır","Muhammed Şengezer","Okan Kocuk","Uğurcan Çakir"],
        "metrics": {
            "goalkeeping__save_percentage": ("Kurtarış %", 0.30),
            "goalkeeping__goals_prevented": ("Önlenen Gol", 0.25),
            "goalkeeping__clean_sheets":    ("Gol Yememe", 0.20),
            "distribution__pass_accuracy":  ("Pas İsabeti", 0.15),
            "goalkeeping__high_claims":     ("Hava Topu", 0.10),
        }
    },
    "Sağ Stoper": {
        "abbr": "RST", "formation_slot": "rcb", "eligible": ["Center Back"],
        "candidates": ["Merih Demiral","Ozan Kabak","Samet Akaydin"],
        "metrics": {
            "defending__interceptions":   ("Top Kapma", 0.23),
            "defending__tackles":         ("Müdahale", 0.19),
            "possession__aerials_won_pct":("Hava Topu %", 0.19),
            "defending__clearances":      ("Uzaklaştırma", 0.16),
            "defending__recoveries":      ("Top Kazanma", 0.13),
            "passing__pass_accuracy":     ("Pas İsabeti", 0.10),
        }
    },
    "Sol Stoper": {
        "abbr": "LST", "formation_slot": "lcb", "eligible": ["Center Back"],
        "candidates": ["Emirhan Topçu","Abdülkerim Bardakci","Adil Demirbağ"],
        "metrics": {
            "defending__interceptions":   ("Top Kapma", 0.23),
            "defending__tackles":         ("Müdahale", 0.19),
            "possession__aerials_won_pct":("Hava Topu %", 0.19),
            "defending__clearances":      ("Uzaklaştırma", 0.16),
            "defending__recoveries":      ("Top Kazanma", 0.13),
            "passing__pass_accuracy":     ("Pas İsabeti", 0.10),
        }
    },
    "Sağ Bek": {
        "abbr": "SBK", "formation_slot": "rb",
        "eligible": ["Right Wing-Back","Right Back","Center Back"],
        "candidates": ["Zeki Çelik"],
        "metrics": {
            "defending__tackles":        ("Müdahale", 0.28),
            "defending__interceptions":  ("Top Kapma", 0.24),
            "passing__xa":               ("Beklenen Asist", 0.24),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.24),
        }
    },
    "Sol Bek": {
        "abbr": "LBK", "formation_slot": "lb",
        "eligible": ["Left Back","Left Wing-Back"],
        "candidates": ["Eren Elmalı","Ferdi Kadıoğlu"],
        "metrics": {
            "passing__xa":               ("Beklenen Asist", 0.31),
            "defending__tackles":        ("Müdahale", 0.23),
            "defending__interceptions":  ("Top Kapma", 0.23),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.23),
        }
    },
    "6 Numara": {
        "abbr": "6", "formation_slot": "dm1", "eligible": ["Defensive Midfielder"],
        "candidates": ["Ismail Yüksek","Melih Kabasakal","Salih Özcan"],
        "metrics": {
            "defending__interceptions":  ("Top Kapma", 0.31),
            "defending__tackles":        ("Müdahale", 0.28),
            "passing__pass_accuracy":    ("Pas İsabeti", 0.24),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.17),
        }
    },
    "8 Numara": {
        "abbr": "8", "formation_slot": "dm2",
        "eligible": ["Defensive Midfielder","Attacking Midfielder"],
        "candidates": ["Orkun Kökcü","Demir Tıknaz","Bartuğ Elmaz"],
        "metrics": {
            "passing__xa":                           ("Beklenen Asist", 0.25),
            "passing__chances_created":              ("Fırsat Yaratma", 0.20),
            "shooting__xg":                          ("Beklenen Gol", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "defending__tackles":                    ("Müdahale", 0.15),
        }
    },
    "Sol Açık": {
        "abbr": "LW", "formation_slot": "lw",
        "eligible": ["Left Winger","Attacking Midfielder","Left Back"],
        "candidates": ["Aral Şimşir","İlhan Fakılı","Barış Alper Yılmaz","Kerem Aktürkoglu","Can Uzun","Yunus Akgün"],
        "metrics": {
            "shooting__xg":                          ("Beklenen Gol", 0.25),
            "passing__xa":                           ("Beklenen Asist", 0.25),
            "possession__dribbles_success_rate":     ("Dribling %", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "shooting__shots_on_target":             ("İsabetli Şut", 0.10),
        }
    },
    "10 Numara": {
        "abbr": "10", "formation_slot": "am",
        "eligible": ["Attacking Midfielder","Defensive Midfielder"],
        "candidates": ["Arda Güler","Can Uzun","İrfan Kahveci","Yunus Akgün"],
        "metrics": {
            "passing__xa":                           ("Beklenen Asist", 0.30),
            "passing__chances_created":              ("Fırsat Yaratma", 0.30),
            "shooting__xg":                          ("Beklenen Gol", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
        }
    },
    "Sağ Açık": {
        "abbr": "RW", "formation_slot": "rw",
        "eligible": ["Right Winger","Attacking Midfielder","Right Wing-Back"],
        "candidates": ["Yunus Akgün","Oğuz Aydın","İrfan Kahveci","Barış Alper Yılmaz"],
        "metrics": {
            "shooting__xg":                          ("Beklenen Gol", 0.25),
            "passing__xa":                           ("Beklenen Asist", 0.20),
            "possession__dribbles_success_rate":     ("Dribling %", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "shooting__shots_on_target":             ("İsabetli Şut", 0.15),
        }
    },
    "Forvet": {
        "abbr": "FW", "formation_slot": "st",
        "eligible": ["Striker","Attacking Midfielder","Left Winger","Right Winger"],
        "candidates": ["Kerem Aktürkoglu","Barış Alper Yılmaz","Deniz Gül"],
        "metrics": {
            "shooting__xg":                          ("Beklenen Gol", 0.30),
            "shooting__goals":                       ("Gol", 0.25),
            "shooting__xgot":                        ("Şut Kalitesi", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.15),
            "possession__aerials_won_pct":           ("Hava Topu %", 0.10),
        }
    },
}

FORMATION_ORDER = [
    "Kaleci",
    "Sağ Bek","Sağ Stoper","Sol Stoper","Sol Bek",
    "6 Numara","8 Numara",
    "Sağ Açık","10 Numara","Sol Açık",
    "Forvet"
]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'weights' not in st.session_state:
    st.session_state.weights = {}

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <div>
        <p class="header-sub">Türkiye Milli Futbol Takımı</p>
        <p class="header-title">⚽ Kadro Optimizasyon Aracı</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("Her mevki için metriklerin ağırlığını ayarla — sistem tüm kombinasyonları değerlendirerek toplam skoru maksimize eden 11'i önerir.")

# ── ÜST BÖLÜM: Ayarlar (sol) + Saha (sağ) ───────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-title">Mevki Ayarları</div>', unsafe_allow_html=True)

    for pos_name in FORMATION_ORDER:
        pos_config = POSITIONS[pos_name]
        with st.expander(f"**{pos_name}** — {pos_config['abbr']}", expanded=False):
            st.caption(f"Aday havuzu: {', '.join(pos_config['candidates'])}")
            weights_override = {}
            metrics_list = list(pos_config['metrics'].items())
            total_default = sum(w for _, (_, w) in metrics_list)
            for col, (label, default_w) in metrics_list:
                pct = int(round((default_w / total_default) * 100))
                new_val = st.slider(label, 0, 100, pct, 5, key=f"{pos_name}_{col}")
                weights_override[col] = new_val / 100.0
            if st.button(f"Bu mevki için uygula", key=f"calc_{pos_name}", type="primary"):
                st.session_state.weights[pos_name] = weights_override

# ── OPTİMİZASYON — tek seferde tüm 11 ───────────────────────────────────────
lineup, all_scores = optimize_lineup(
    POSITIONS,
    FORMATION_ORDER,
    st.session_state.weights,
    df
)

with right_col:
    st.markdown('<div class="section-title">Önerilen 11 — Formasyon</div>', unsafe_allow_html=True)

    if lineup:
        W, H = 400, 580

        formation_slots = {
            "Kaleci":     (200, 520),
            "Sağ Bek":    (340, 400),
            "Sağ Stoper": (250, 400),
            "Sol Stoper": (150, 400),
            "Sol Bek":    ( 60, 400),
            "6 Numara":   (140, 290),
            "8 Numara":   (260, 290),
            "Sağ Açık":   (340, 170),
            "10 Numara":  (200, 170),
            "Sol Açık":   ( 60, 170),
            "Forvet":     (200,  55),
        }

        def player_card_svg(cx, cy, pos_abbr, player_name, score, is_best=True):
            short_name = player_name.split()[-1] if player_name else "—"
            score_txt  = str(score) if score else "—"
            card_w, card_h = 76, 52
            x0 = cx - card_w / 2
            y0 = cy - card_h / 2
            badge_color = "#e63946" if is_best else "#4a5568"
            score_color = "#8bc34a" if is_best else "#9ca3af"
            return f"""
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{card_w}" height="{card_h}"
        rx="6" fill="rgba(8,13,26,0.82)" stroke="{badge_color}" stroke-width="1.4"/>
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{card_w}" height="14"
        rx="6" fill="{badge_color}"/>
  <rect x="{x0:.1f}" y="{y0+8:.1f}" width="{card_w}" height="6" fill="{badge_color}"/>
  <text x="{cx:.1f}" y="{y0+10.5:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Barlow Condensed,sans-serif" font-size="8" font-weight="700"
        fill="#ffffff" letter-spacing="0.5">{pos_abbr}</text>
  <text x="{cx:.1f}" y="{y0+27:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Inter,sans-serif" font-size="9.5" font-weight="600"
        fill="#ffffff">{short_name}</text>
  <text x="{cx:.1f}" y="{y0+42:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Barlow Condensed,sans-serif" font-size="11" font-weight="700"
        fill="{score_color}">{score_txt}</text>"""

        pitch_lines = f"""
  <rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="url(#grass)"/>
  <rect x="20" y="15" width="{W-40}" height="{H-30}" rx="4"
        fill="none" stroke="#2d6a2d" stroke-width="1.5"/>
  <line x1="20" y1="{H//2}" x2="{W-20}" y2="{H//2}" stroke="#2d6a2d" stroke-width="1.2"/>
  <circle cx="{W//2}" cy="{H//2}" r="38" fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <circle cx="{W//2}" cy="{H//2}" r="2.5" fill="#2d6a2d"/>
  <rect x="105" y="15" width="190" height="72" fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <rect x="148" y="15" width="104" height="30" fill="none" stroke="#2d6a2d" stroke-width="1.0"/>
  <rect x="105" y="{H-87}" width="190" height="72" fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <rect x="148" y="{H-45}" width="104" height="30" fill="none" stroke="#2d6a2d" stroke-width="1.0"/>
  <circle cx="{W//2}" cy="56" r="2" fill="#2d6a2d"/>
  <circle cx="{W//2}" cy="{H-56}" r="2" fill="#2d6a2d"/>"""

        cards_svg = ""
        for pos_name, (cx, cy) in formation_slots.items():
            player = lineup.get(pos_name)
            abbr   = POSITIONS[pos_name]["abbr"]
            if player:
                cards_svg += player_card_svg(cx, cy, abbr, player['name'], player['score'], is_best=True)
            else:
                cards_svg += player_card_svg(cx, cy, abbr, "—", None, is_best=False)

        svg_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  svg {{ width:100%; max-width:420px; display:block; margin:0 auto; border-radius:10px; }}
</style></head><body>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grass" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"  stop-color="#1a3d1a"/>
      <stop offset="50%" stop-color="#1f4a1f"/>
      <stop offset="100%" stop-color="#1a3d1a"/>
    </linearGradient>
  </defs>
  {pitch_lines}
  {cards_svg}
</svg></body></html>"""

        components.html(svg_html, height=H + 20, scrolling=False)

# ── ALT BÖLÜM: Kadro + Alternatifler ────────────────────────────────────────
st.divider()

if lineup:
    detail_col, alt_col = st.columns([1, 1], gap="large")

    with detail_col:
        st.markdown('<div class="section-title">Oyuncu Detayları</div>', unsafe_allow_html=True)
        for pos_name in FORMATION_ORDER:
            player = lineup.get(pos_name)
            if not player:
                continue
            st.markdown(f"""
            <div class="pos-card" style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="pos-card-title">{pos_name}</div>
                    <div class="player-name">{player['name']}</div>
                    <div class="player-meta">{player['team']} · {player['league']}</div>
                </div>
                <div class="score-badge">{player['score']}</div>
            </div>
            """, unsafe_allow_html=True)

    with alt_col:
        st.markdown('<div class="section-title">Alternatif Oyuncular</div>', unsafe_allow_html=True)
        for pos_name in FORMATION_ORDER:
            scores = all_scores.get(pos_name, [])
            if len(scores) > 1:
                selected_name = lineup.get(pos_name, {}).get('name', '')
                with st.expander(f"{pos_name} alternatifleri"):
                    rows = []
                    for d in scores[:5]:
                        secili = "✓" if d['name'] == selected_name else ""
                        rows.append({
                            "": secili,
                            "Oyuncu": d['name'],
                            "Skor": d['score'],
                            "Takım": d['team'],
                            "Lig": d['league'],
                        })
                    alt_df = pd.DataFrame(rows)
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Veri: FotMob · Metodoloji: Wyscout Index & Apunts (2024) · Geliştirici: M. Enes Şahin")
