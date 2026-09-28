import streamlit as st
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Barlow+Condensed:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080d1a; color: #e8eaf0; }
h1,h2,h3 { font-family: 'Barlow Condensed', sans-serif; }

.header-band {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5e 50%, #0d1b3e 100%);
    border-bottom: 3px solid #e63946;
    padding: 1.2rem 2rem; margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 1.2rem;
}
.header-title { font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:800; color:#fff; margin:0; }
.header-sub { font-size:0.75rem; color:#8899bb; letter-spacing:1.5px; text-transform:uppercase; margin:0; }
.header-badge { background:#e63946; color:#fff; font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.8rem; padding:0.2rem 0.6rem; border-radius:4px; letter-spacing:1px; }

.section-title {
    font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; font-weight:700;
    color:#8899bb; text-transform:uppercase; letter-spacing:2px;
    border-left:3px solid #e63946; padding-left:0.7rem; margin:1.2rem 0 0.8rem 0;
}

.player-card {
    background:#141f35; border:2px solid #1e2d4a; border-radius:8px;
    padding:0.65rem 0.8rem; position:relative; margin-bottom:0.3rem;
}
.player-card.selected { border-color:#e63946; background:rgba(230,57,70,0.07); }
.player-card.best-only { border-color:#8bc34a; }
.player-card-name { font-weight:600; font-size:0.88rem; color:#fff; }
.player-card-team { font-size:0.7rem; color:#6b7fa3; margin-top:0.1rem; }
.player-score { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.25rem; position:absolute; right:0.8rem; top:50%; transform:translateY(-50%); }
.sg { color:#8bc34a; } .sy { color:#ffc107; } .sr { color:#e63946; }
.badge { font-size:0.58rem; font-weight:700; padding:0.1rem 0.3rem; border-radius:3px; margin-left:0.35rem; vertical-align:middle; letter-spacing:0.5px; }
.badge-g { background:#8bc34a; color:#000; }
.badge-r { background:#e63946; color:#fff; }

.stat-row { display:flex; align-items:center; gap:0.4rem; margin:0.2rem 0; font-size:0.68rem; }
.stat-label { color:#6b7fa3; min-width:88px; }
.stat-bar-bg { flex:1; height:3px; background:#1e2d4a; border-radius:2px; }
.stat-bar-fill { height:3px; border-radius:2px; background:#e63946; }

.lineup-row { display:flex; align-items:center; justify-content:space-between; background:#0f1626; border:1px solid #1e2d4a; border-radius:8px; padding:0.55rem 0.9rem; margin-bottom:0.4rem; }
.lr-pos { color:#e63946; font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.85rem; min-width:46px; }
.lr-name { font-weight:600; font-size:0.88rem; color:#fff; }
.lr-team { font-size:0.68rem; color:#6b7fa3; }
.lr-score { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.1rem; }

.pitch-wrapper { background:linear-gradient(180deg,#1a3a1a,#1e4420,#1a3a1a); border:2px solid #2d5a2d; border-radius:12px; overflow:hidden; }
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

def minmax_normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5]*len(series), index=series.index)
    return (series - mn) / (mx - mn)

@st.cache_data
def build_normalized(df):
    norm = pd.DataFrame()
    norm['name'] = df['name']
    norm['position_primary'] = df['position_primary']
    norm['team'] = df['team']
    norm['league'] = df['league']
    norm['age'] = df['age']
    pos_cols = [
        'goalkeeping__save_percentage','goalkeeping__goals_prevented','goalkeeping__clean_sheets',
        'goalkeeping__high_claims','distribution__pass_accuracy','defending__interceptions',
        'defending__tackles','defending__clearances','defending__recoveries',
        'defending__possession_won_final_3rd','possession__aerials_won_pct','possession__duels_won_pct',
        'possession__touches_in_opposition_box','possession__dribbles_success_rate',
        'passing__pass_accuracy','passing__xa','passing__chances_created',
        'shooting__xg','shooting__goals','shooting__shots_on_target','shooting__xgot',
    ]
    neg_cols = ['defending__dribbled_past','defending__fouls_committed','goalkeeping__error_led_to_goal']
    for col in pos_cols:
        if col in df.columns:
            norm[col] = minmax_normalize(df[col].fillna(df[col].median()))
    for col in neg_cols:
        if col in df.columns:
            norm[col] = 1 - minmax_normalize(df[col].fillna(df[col].median()))
    return norm

norm_df = build_normalized(df)

POSITIONS = {
    "Kaleci": {
        "abbr": "KAL",
        "candidates": ["Altay Bayındır","Muhammed Şengezer","Okan Kocuk","Uğurcan Çakir"],
        "metrics": {
            "goalkeeping__save_percentage": ("Kurtarış %", 0.30),
            "goalkeeping__goals_prevented": ("Önlenen Gol", 0.25),
            "goalkeeping__clean_sheets": ("Gol Yememe", 0.20),
            "distribution__pass_accuracy": ("Pas İsabeti", 0.15),
            "goalkeeping__high_claims": ("Hava Topu", 0.10),
        }
    },
    "Sağ Stoper": {
        "abbr": "RST",
        "candidates": ["Merih Demiral","Ozan Kabak","Samet Akaydin"],
        "metrics": {
            "defending__interceptions": ("Top Kapma", 0.23),
            "defending__tackles": ("Müdahale", 0.19),
            "possession__aerials_won_pct": ("Hava Topu %", 0.19),
            "defending__clearances": ("Uzaklaştırma", 0.16),
            "defending__recoveries": ("Top Kazanma", 0.13),
            "passing__pass_accuracy": ("Pas İsabeti", 0.10),
        }
    },
    "Sol Stoper": {
        "abbr": "LST",
        "candidates": ["Emirhan Topçu","Abdülkerim Bardakci","Adil Demirbağ"],
        "metrics": {
            "defending__interceptions": ("Top Kapma", 0.23),
            "defending__tackles": ("Müdahale", 0.19),
            "possession__aerials_won_pct": ("Hava Topu %", 0.19),
            "defending__clearances": ("Uzaklaştırma", 0.16),
            "defending__recoveries": ("Top Kazanma", 0.13),
            "passing__pass_accuracy": ("Pas İsabeti", 0.10),
        }
    },
    "Sağ Bek": {
        "abbr": "SBK",
        "candidates": ["Zeki Çelik"],
        "metrics": {
            "defending__tackles": ("Müdahale", 0.28),
            "defending__interceptions": ("Top Kapma", 0.24),
            "passing__xa": ("Beklenen Asist", 0.24),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.24),
        }
    },
    "Sol Bek": {
        "abbr": "LBK",
        "candidates": ["Eren Elmalı","Ferdi Kadıoğlu"],
        "metrics": {
            "passing__xa": ("Beklenen Asist", 0.31),
            "defending__tackles": ("Müdahale", 0.23),
            "defending__interceptions": ("Top Kapma", 0.23),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.23),
        }
    },
    "6 Numara": {
        "abbr": "6",
        "candidates": ["Ismail Yüksek","Melih Kabasakal","Salih Özcan"],
        "metrics": {
            "defending__interceptions": ("Top Kapma", 0.31),
            "defending__tackles": ("Müdahale", 0.28),
            "passing__pass_accuracy": ("Pas İsabeti", 0.24),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.17),
        }
    },
    "8 Numara": {
        "abbr": "8",
        "candidates": ["Orkun Kökcü","Demir Tıknaz","Bartuğ Elmaz"],
        "metrics": {
            "passing__xa": ("Beklenen Asist", 0.25),
            "passing__chances_created": ("Fırsat Yaratma", 0.20),
            "shooting__xg": ("Beklenen Gol", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "defending__tackles": ("Müdahale", 0.15),
        }
    },
    "Sol Açık": {
        "abbr": "LW",
        "candidates": ["Aral Şimşir","İlhan Fakılı"],
        "metrics": {
            "shooting__xg": ("Beklenen Gol", 0.25),
            "passing__xa": ("Beklenen Asist", 0.25),
            "possession__dribbles_success_rate": ("Dribling %", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "shooting__shots_on_target": ("İsabetli Şut", 0.10),
        }
    },
    "10 Numara": {
        "abbr": "10",
        "candidates": ["Arda Güler","Can Uzun"],
        "metrics": {
            "passing__xa": ("Beklenen Asist", 0.30),
            "passing__chances_created": ("Fırsat Yaratma", 0.30),
            "shooting__xg": ("Beklenen Gol", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
        }
    },
    "Sağ Açık": {
        "abbr": "RW",
        "candidates": ["Yunus Akgün","Oğuz Aydın","İrfan Kahveci"],
        "metrics": {
            "shooting__xg": ("Beklenen Gol", 0.25),
            "passing__xa": ("Beklenen Asist", 0.20),
            "possession__dribbles_success_rate": ("Dribling %", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
            "shooting__shots_on_target": ("İsabetli Şut", 0.15),
        }
    },
    "Forvet": {
        "abbr": "FW",
        "candidates": ["Kerem Aktürkoglu","Barış Alper Yılmaz","Deniz Gül"],
        "metrics": {
            "shooting__xg": ("Beklenen Gol", 0.30),
            "shooting__goals": ("Gol", 0.25),
            "shooting__xgot": ("Şut Kalitesi", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.15),
            "possession__aerials_won_pct": ("Hava Topu %", 0.10),
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

FORMATION_ROWS_SVG = {
    "fw":  ["Forvet"],
    "att": ["Sağ Açık","10 Numara","Sol Açık"],
    "mid": ["6 Numara","8 Numara"],
    "def": ["Sağ Bek","Sağ Stoper","Sol Stoper","Sol Bek"],
    "gk":  ["Kaleci"],
}

# ── SKOR ─────────────────────────────────────────────────────────────────────
def compute_score(player_row, metrics_weights):
    score = 0.0
    for col, (label, w) in metrics_weights.items():
        val = player_row.get(col, 0.5)
        if pd.isna(val): val = 0.5
        score += val * w
    return round(score * 100, 1)

def get_all_scores(pos_name, pos_config, norm_df):
    eligible = norm_df[norm_df['name'].isin(pos_config['candidates'])].copy()
    if eligible.empty:
        return []
    metrics = pos_config['metrics']
    total_w = sum(w for _, w in metrics.values())
    norm_m = {col: (lbl, w/total_w) for col, (lbl, w) in metrics.items()} if total_w > 0 else metrics
    results = []
    for _, row in eligible.iterrows():
        s = compute_score(row, norm_m)
        metric_vals = {}
        for col in norm_m:
            val = row.get(col, 0.5)
            metric_vals[col] = float(val) if not pd.isna(val) else 0.5
        results.append({"name": row['name'], "score": s, "team": row['team'],
                        "league": row['league'], "position": row['position_primary'], "metrics": metric_vals})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def sc_cls(s):
    if s >= 65: return "sg"
    if s >= 45: return "sy"
    return "sr"

# ── SVG SAHA ─────────────────────────────────────────────────────────────────
def build_pitch_svg(lineup, manual_picks):
    W, H = 400, 530
    row_y = {"fw": 65, "att": 160, "mid": 258, "def": 358, "gk": 450}

    def row_xs(n):
        margin = 40
        if n == 1: return [W/2]
        step = (W - 2*margin) / (n-1)
        return [margin + i*step for i in range(n)]

    lines = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;display:block">',
        # Pitch background
        '<defs>'
        '<linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#1a3a1a"/>'
        '<stop offset="50%" stop-color="#1e4a20"/>'
        '<stop offset="100%" stop-color="#1a3a1a"/>'
        '</linearGradient>'
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="url(#pitchGrad)"/>',
        # Pitch stripes
        *[f'<rect x="0" y="{30+i*60}" width="{W}" height="30" fill="rgba(0,0,0,0.06)"/>' for i in range(8)],
        # Outer boundary
        f'<rect x="16" y="10" width="{W-32}" height="{H-20}" rx="3" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>',
        # Centre line
        f'<line x1="16" y1="{H//2}" x2="{W-16}" y2="{H//2}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>',
        # Centre circle
        f'<circle cx="{W//2}" cy="{H//2}" r="50" fill="none" stroke="rgba(255,255,255,0.13)" stroke-width="1"/>',
        f'<circle cx="{W//2}" cy="{H//2}" r="3" fill="rgba(255,255,255,0.35)"/>',
        # Penalty areas — top (attack)
        f'<rect x="100" y="10" width="{W-200}" height="100" rx="2" fill="none" stroke="rgba(255,255,255,0.13)" stroke-width="1"/>',
        f'<rect x="148" y="10" width="{W-296}" height="44" rx="2" fill="none" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>',
        # Penalty areas — bottom (defence/keeper)
        f'<rect x="100" y="{H-110}" width="{W-200}" height="100" rx="2" fill="none" stroke="rgba(255,255,255,0.13)" stroke-width="1"/>',
        f'<rect x="148" y="{H-54}" width="{W-296}" height="44" rx="2" fill="none" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>',
    ]

    for row_key, pos_list in FORMATION_ROWS_SVG.items():
        y = row_y[row_key]
        xs = row_xs(len(pos_list))
        for i, pos_name in enumerate(pos_list):
            x = xs[i]
            p = lineup.get(pos_name)
            abbr = POSITIONS[pos_name]["abbr"]
            is_manual = pos_name in manual_picks
            if p:
                parts = p["name"].split()
                short = parts[-1] if len(parts) > 1 else parts[0]
                sc = p["score"]
                sc_col = "#8bc34a" if sc >= 65 else ("#ffc107" if sc >= 45 else "#e63946")
                star = "★" if is_manual else ""
            else:
                short, sc, sc_col, star = "—", "", "#555", ""

            bw, bh = 76, 58
            bx, by = x - bw/2, y - bh/2

            border_col = "#e63946" if not is_manual else "#ffd700"

            lines.append(f'''
            <g>
              <rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh}"
                    rx="7" fill="rgba(8,13,26,0.88)" stroke="{border_col}" stroke-width="2"/>
              <text x="{x:.1f}" y="{by+14:.1f}" text-anchor="middle"
                    font-family="Barlow Condensed,sans-serif" font-size="9.5" font-weight="700"
                    fill="{border_col}" letter-spacing="0.5">{abbr}{star}</text>
              <text x="{x:.1f}" y="{by+30:.1f}" text-anchor="middle"
                    font-family="Inter,sans-serif" font-size="10.5" font-weight="700"
                    fill="#ffffff">{short}</text>
              <text x="{x:.1f}" y="{by+47:.1f}" text-anchor="middle"
                    font-family="Barlow Condensed,sans-serif" font-size="14" font-weight="700"
                    fill="{sc_col}">{sc}</text>
            </g>''')

    lines.append("</svg>")
    return "\n".join(lines)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'manual_picks' not in st.session_state:
    st.session_state.manual_picks = {}

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <div style="font-size:2.8rem;line-height:1">🇹🇷</div>
    <div>
        <p class="header-sub">Türkiye Milli Futbol Takımı · 1-4-2-3-1</p>
        <p class="header-title">Kadro Optimizasyon Aracı</p>
    </div>
    <div style="margin-left:auto"><span class="header-badge">2024–25 VERİSİ</span></div>
</div>
""", unsafe_allow_html=True)

# Precompute all scores
all_pos_scores = {}
for pos_name in FORMATION_ORDER:
    all_pos_scores[pos_name] = get_all_scores(pos_name, POSITIONS[pos_name], norm_df)

# Build lineup
lineup = {}
used_names = set()
for pos_name in FORMATION_ORDER:
    scores = all_pos_scores.get(pos_name, [])
    if not scores: continue
    manual = st.session_state.manual_picks.get(pos_name, None)
    if manual:
        chosen = next((p for p in scores if p["name"] == manual), scores[0])
    else:
        chosen = next((p for p in scores if p["name"] not in used_names), scores[0])
    lineup[pos_name] = chosen
    used_names.add(chosen["name"])

# ── ANA LAYOUT ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# ── SOL: OYUNCU KARTI SEÇİMİ ─────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-title">Oyuncu Seçimi</div>', unsafe_allow_html=True)
    st.caption("Sistem en yüksek skoru alan oyuncuyu otomatik seçer. Farklı oyuncuyu seçmek için butona tıkla.")

    for pos_name in FORMATION_ORDER:
        pos_config = POSITIONS[pos_name]
        scores = all_pos_scores.get(pos_name, [])
        if not scores: continue

        best_auto_name = scores[0]["name"]
        manual = st.session_state.manual_picks.get(pos_name, None)
        selected_name = manual if manual else best_auto_name
        sel_player = lineup.get(pos_name)
        sel_score = sel_player["score"] if sel_player else 0

        with st.expander(
            f"**{pos_config['abbr']}** · {pos_name}  —  **{selected_name}**  `{sel_score}`",
            expanded=False
        ):
            for p in scores:
                is_sel = p["name"] == selected_name
                is_best = p["name"] == best_auto_name

                badges = ""
                if is_best:
                    badges += '<span class="badge badge-g">EN İYİ</span>'
                if is_sel and manual:
                    badges += '<span class="badge badge-r">SEÇİLİ</span>'

                card_cls = "player-card selected" if is_sel else ("player-card best-only" if is_best else "player-card")

                bars_html = ""
                for col, val in list(p["metrics"].items())[:3]:
                    lbl = pos_config["metrics"][col][0]
                    pct = int(val * 100)
                    bars_html += (
                        f'<div class="stat-row">'
                        f'<span class="stat-label">{lbl}</span>'
                        f'<div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct}%"></div></div>'
                        f'<span style="color:#8bc34a;font-size:0.66rem;min-width:26px;text-align:right">{pct}</span>'
                        f'</div>'
                    )

                s = p["score"]
                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="player-card-name">{p['name']}{badges}</div>
                    <div class="player-card-team">{p['team']} · {p['league']}</div>
                    {bars_html}
                    <span class="player-score {sc_cls(s)}">{s}</span>
                </div>
                """, unsafe_allow_html=True)

                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn2:
                    if is_sel and manual:
                        if st.button("Otomatiğe Dön", key=f"auto_{pos_name}_{p['name']}", use_container_width=True):
                            st.session_state.manual_picks.pop(pos_name, None)
                            st.rerun()
                    elif not is_sel:
                        if st.button("Seç", key=f"pick_{pos_name}_{p['name']}", type="primary", use_container_width=True):
                            st.session_state.manual_picks[pos_name] = p["name"]
                            st.rerun()

# ── SAĞ: SAHA + ÖZET ─────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-title">Önerilen 11 · 1-4-2-3-1</div>', unsafe_allow_html=True)

    # SVG Saha
    svg_html = build_pitch_svg(lineup, st.session_state.manual_picks)
    st.markdown(f'<div class="pitch-wrapper">{svg_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # Kadro Listesi
    st.markdown('<div class="section-title">Kadro</div>', unsafe_allow_html=True)
    for pos_name in FORMATION_ORDER:
        p = lineup.get(pos_name)
        if not p: continue
        s = p["score"]
        abbr = POSITIONS[pos_name]["abbr"]
        is_manual = pos_name in st.session_state.manual_picks
        manual_dot = ' <span style="color:#ffd700;font-size:0.7rem">★</span>' if is_manual else ""
        st.markdown(f"""
        <div class="lineup-row">
            <span class="lr-pos">{abbr}</span>
            <div><div class="lr-name">{p['name']}{manual_dot}</div>
                 <div class="lr-team">{p['team']}</div></div>
            <span class="lr-score {sc_cls(s)}">{s}</span>
        </div>
        """, unsafe_allow_html=True)

    if lineup:
        avg = round(np.mean([p["score"] for p in lineup.values()]), 1)
        manual_count = len(st.session_state.manual_picks)
        auto_info = f"{manual_count} manuel · {11-manual_count} otomatik" if manual_count else "Tümü otomatik"
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;background:#0f1626;border:1px solid #1e2d4a;
             border-radius:10px;padding:1rem">
            <div style="color:#6b7fa3;font-size:0.72rem;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.3rem">Takım Ortalama Skor</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;font-weight:800;color:#8bc34a;line-height:1">{avg}</div>
            <div style="color:#6b7fa3;font-size:0.68rem;margin-top:0.3rem">{auto_info}</div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Veri: FotMob · Metodoloji: Wyscout Index & Apunts (2024) · Geliştirici: M. Enes Şahin")
