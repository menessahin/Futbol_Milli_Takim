import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Türkiye Milli Takım | Kadro Optimizasyonu",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0a0f1e; color: #e8eaf0; }

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.5px; }

/* Header */
.header-band {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5e 50%, #0d1b3e 100%);
    border-bottom: 2px solid #e63946;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-title { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #fff; margin: 0; }
.header-sub { font-size: 0.8rem; color: #8899bb; letter-spacing: 1px; text-transform: uppercase; margin: 0; }

/* Pozisyon kartları */
.pos-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.pos-card:hover { border-color: #e63946; }
.pos-card-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e63946;
    margin-bottom: 0.25rem;
}
.player-name {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
}
.player-meta {
    font-size: 0.75rem;
    color: #6b7fa3;
    margin-top: 0.2rem;
}
.score-badge {
    background: #e63946;
    color: white;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    float: right;
}

/* Formasyon sahası */
.pitch {
    background: linear-gradient(180deg, #1a3a1a 0%, #1e4420 50%, #1a3a1a 100%);
    border: 2px solid #2d5a2d;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    position: relative;
    min-height: 500px;
}
.pitch-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 0.5rem 0;
}
.pitch-player {
    background: rgba(230, 57, 70, 0.15);
    border: 1.5px solid #e63946;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    text-align: center;
    min-width: 90px;
    font-size: 0.72rem;
}
.pitch-player-name { font-weight: 600; color: #fff; font-size: 0.78rem; }
.pitch-player-pos { color: #e63946; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; }
.pitch-player-score { color: #8bc34a; font-weight: 700; font-size: 0.85rem; }

/* Slider override */
.stSlider > div > div > div { background: #e63946 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #6b7fa3; }
.stTabs [aria-selected="true"] { color: #e63946 !important; }

/* Expander */
.streamlit-expanderHeader { background: #111827 !important; color: #e8eaf0 !important; border-radius: 6px; }

/* Table */
.stDataFrame { background: #111827; }

/* Metric */
[data-testid="stMetricValue"] { color: #e63946; font-family: 'Barlow Condensed', sans-serif; font-size: 2rem !important; }

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #8899bb;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-left: 3px solid #e63946;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── VERİ ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("milli_takim_stats_full.xlsx")
    # Altay Bayındır'ın verileri eski sezona ait — sayısal kolonlar NaN yapıldı
    altay_idx = df[df['name'] == 'Altay Bayındır'].index
    if len(altay_idx) > 0:
        numeric_cols = df.select_dtypes(include='number').columns
        df.loc[altay_idx, numeric_cols] = None
    return df

df = load_data()

# ── NORMALİZASYON ────────────────────────────────────────────────────────────
def minmax_normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def normalize_inverse(series):
    """Yüksek = kötü metrikleri ters normalize et"""
    return 1 - minmax_normalize(series)

@st.cache_data
def build_normalized(df):
    norm = pd.DataFrame()
    norm['name'] = df['name']
    norm['position_primary'] = df['position_primary']
    norm['team'] = df['team']
    norm['league'] = df['league']
    norm['age'] = df['age']

    # Pozitif metrikler (yüksek = iyi)
    pos_cols = [
        'goalkeeping__save_percentage', 'goalkeeping__goals_prevented',
        'goalkeeping__clean_sheets', 'goalkeeping__high_claims',
        'distribution__pass_accuracy',
        'defending__interceptions', 'defending__tackles', 'defending__clearances',
        'defending__recoveries', 'defending__possession_won_final_3rd',
        'possession__aerials_won_pct', 'possession__duels_won_pct',
        'possession__touches_in_opposition_box', 'possession__dribbles_success_rate',
        'passing__pass_accuracy', 'passing__xa', 'passing__chances_created',
        'shooting__xg', 'shooting__goals', 'shooting__shots_on_target', 'shooting__xgot',
    ]
    # Negatif metrikler (yüksek = kötü) → ters normalize
    neg_cols = [
        'defending__dribbled_past', 'defending__fouls_committed',
        'goalkeeping__error_led_to_goal',
    ]

    for col in pos_cols:
        if col in df.columns:
            norm[col] = minmax_normalize(df[col].fillna(df[col].median()))

    for col in neg_cols:
        if col in df.columns:
            norm[col] = normalize_inverse(df[col].fillna(df[col].median()))

    return norm

norm_df = build_normalized(df)

# ── FORMASYON & POZİSYON TANIMI ─────────────────────────────────────────────
POSITIONS = {
    "Kaleci": {
        "abbr": "KAL",
        "formation_slot": "gk",
        "eligible": ["Keeper"],
        "candidates": ["Altay Bayındır", "Muhammed Şengezer", "Okan Kocuk", "Uğurcan Çakir"],
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
        "formation_slot": "rcb",
        "eligible": ["Center Back"],
        "candidates": ["Merih Demiral", "Ozan Kabak", "Samet Akaydin"],
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
        "formation_slot": "lcb",
        "eligible": ["Center Back"],
        "candidates": ["Emirhan Topçu", "Abdülkerim Bardakci", "Adil Demirbağ"],
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
        "formation_slot": "rb",
        "eligible": ["Right Wing-Back", "Right Back", "Center Back"],
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
        "formation_slot": "lb",
        "eligible": ["Left Back", "Left Wing-Back"],
        "candidates": ["Eren Elmalı", "Ferdi Kadıoğlu"],
        "metrics": {
            "passing__xa": ("Beklenen Asist", 0.31),
            "defending__tackles": ("Müdahale", 0.23),
            "defending__interceptions": ("Top Kapma", 0.23),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.23),
        }
    },
    "6 Numara": {
        "abbr": "6",
        "formation_slot": "dm1",
        "eligible": ["Defensive Midfielder"],
        "candidates": ["Ismail Yüksek", "Melih Kabasakal", "Salih Özcan"],
        "metrics": {
            "defending__interceptions": ("Top Kapma", 0.31),
            "defending__tackles": ("Müdahale", 0.28),
            "passing__pass_accuracy": ("Pas İsabeti", 0.24),
            "possession__duels_won_pct": ("İkili Kazanma %", 0.17),
        }
    },
    "8 Numara": {
        "abbr": "8",
        "formation_slot": "dm2",
        "eligible": ["Defensive Midfielder", "Attacking Midfielder"],
        "candidates": ["Orkun Kökcü", "Demir Tıknaz", "Bartuğ Elmaz"],
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
        "formation_slot": "lw",
        "eligible": ["Left Winger", "Attacking Midfielder", "Left Back"],
        "candidates": ["Aral Şimşir", "İlhan Fakılı"],
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
        "formation_slot": "am",
        "eligible": ["Attacking Midfielder", "Defensive Midfielder"],
        "candidates": ["Arda Güler", "Can Uzun"],
        "metrics": {
            "passing__xa": ("Beklenen Asist", 0.30),
            "passing__chances_created": ("Fırsat Yaratma", 0.30),
            "shooting__xg": ("Beklenen Gol", 0.20),
            "possession__touches_in_opposition_box": ("Rakip Ceza Dokunuş", 0.20),
        }
    },
    "Sağ Açık": {
        "abbr": "RW",
        "formation_slot": "rw",
        "eligible": ["Right Winger", "Attacking Midfielder", "Right Wing-Back"],
        "candidates": ["Yunus Akgün", "Oğuz Aydın", "İrfan Kahveci"],
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
        "formation_slot": "st",
        "eligible": ["Striker", "Attacking Midfielder", "Left Winger", "Right Winger"],
        "candidates": ["Kerem Aktürkoglu", "Barış Alper Yılmaz", "Deniz Gül"],
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
    "Sağ Bek", "Sağ Stoper", "Sol Stoper", "Sol Bek",
    "6 Numara", "8 Numara",
    "Sağ Açık", "10 Numara", "Sol Açık",
    "Forvet"
]

# ── SKOR HESAPLAMA ────────────────────────────────────────────────────────────
def compute_score(player_row, metrics_weights):
    score = 0.0
    for col, (label, w) in metrics_weights.items():
        val = player_row.get(col, 0.5)
        if pd.isna(val):
            val = 0.5
        score += val * w
    return round(score * 100, 1)

def get_best_player(pos_name, pos_config, weights_override, used_names, norm_df):
    # Aday havuzu artık geniş pozisyon kategorisine değil, elle atanmış oyuncu
    # listesine (candidates) göre belirleniyor — bkz. proje notu.
    eligible = norm_df[norm_df['name'].isin(pos_config['candidates'])].copy()
    eligible = eligible[~eligible['name'].isin(used_names)]
    if eligible.empty:
        return None, 0

    metrics = {}
    for col, (label, default_w) in pos_config['metrics'].items():
        metrics[col] = (label, weights_override.get(col, default_w))

    # Ağırlıkları normalize et (toplamı 1'e tamamla)
    total_w = sum(w for _, w in metrics.values())
    if total_w > 0:
        metrics = {col: (lbl, w / total_w) for col, (lbl, w) in metrics.items()}

    scores = []
    for _, row in eligible.iterrows():
        s = compute_score(row, metrics)
        scores.append((row['name'], s, row['team'], row['league'], row['position_primary']))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0], scores

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'weights' not in st.session_state:
    st.session_state.weights = {}
if 'lineup' not in st.session_state:
    st.session_state.lineup = {}

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <div>
        <p class="header-sub">Türkiye Milli Futbol Takımı</p>
        <p class="header-title">⚽ Kadro Optimizasyon Aracı</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Her mevki için metriklerin ağırlığını ayarla — sistem en yüksek skoru alan oyuncuyu otomatik olarak önerir.
""")

# ── ANA LAYOUT ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-title">Mevki Ayarları</div>', unsafe_allow_html=True)

    lineup = {}
    used_names = set()
    all_scores = {}

    unique_positions = FORMATION_ORDER

    for pos_name in unique_positions:
        pos_config = POSITIONS[pos_name]

        with st.expander(f"**{pos_name}** — {pos_config['abbr']}", expanded=False):
            st.caption(f"Aday havuzu: {', '.join(pos_config['candidates'])}")

            weights_override = {}
            metrics_list = list(pos_config['metrics'].items())
            total_default = sum(w for _, (_, w) in metrics_list)

            for col, (label, default_w) in metrics_list:
                pct = int(round((default_w / total_default) * 100))
                new_val = st.slider(
                    label,
                    min_value=0,
                    max_value=100,
                    value=pct,
                    step=5,
                    key=f"{pos_name}_{col}"
                )
                weights_override[col] = new_val / 100.0

            # Hesapla butonu
            if st.button(f"Bu mevki için hesapla", key=f"calc_{pos_name}", type="primary"):
                st.session_state.weights[pos_name] = weights_override

        # Skoru hesapla (slider değişse bile)
        current_weights = st.session_state.weights.get(pos_name, {})
        best, scores = get_best_player(pos_name, pos_config, current_weights, used_names, norm_df)

        if best:
            lineup[pos_name] = best
            used_names.add(best[0])
            all_scores[pos_name] = scores

    st.session_state.lineup = lineup

with right_col:
    st.markdown('<div class="section-title">Önerilen 11</div>', unsafe_allow_html=True)

    if lineup:
        # Formasyon görünümü
        formation_rows = {
            "gk":  ["Kaleci"],
            "def": ["Sağ Bek", "Sağ Stoper", "Sol Stoper", "Sol Bek"],
            "mid": ["6 Numara", "8 Numara"],
            "att": ["Sağ Açık", "10 Numara", "Sol Açık"],
            "fw":  ["Forvet"],
        }

        st.markdown('<div class="pitch">', unsafe_allow_html=True)
        for row_key, pos_list in formation_rows.items():
            cols = st.columns(len(pos_list))
            for i, pos_name in enumerate(pos_list):
                player = lineup.get(pos_name)
                with cols[i]:
                    if player:
                        st.markdown(f"""
                        <div class="pitch-player">
                            <div class="pitch-player-pos">{POSITIONS[pos_name]['abbr']}</div>
                            <div class="pitch-player-name">{player[0].split()[0]}</div>
                            <div class="pitch-player-score">{player[1]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pitch-player" style="opacity:0.4">
                            <div class="pitch-player-pos">{POSITIONS[pos_name]['abbr']}</div>
                            <div class="pitch-player-name">—</div>
                            <div class="pitch-player-score">-</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Kadro listesi
        st.markdown('<div class="section-title">Oyuncu Detayları</div>', unsafe_allow_html=True)

        for pos_name, player in lineup.items():
            name, score, team, league, primary_pos = player
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="pos-card">
                    <div class="pos-card-title">{pos_name}</div>
                    <div class="player-name">{name}</div>
                    <div class="player-meta">{team} · {league} · {primary_pos}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.metric("Skor", f"{score}")

        # Alternatifler
        st.markdown('<div class="section-title">Alternatif Oyuncular</div>', unsafe_allow_html=True)
        for pos_name, scores in all_scores.items():
            if scores and len(scores) > 1:
                with st.expander(f"{pos_name} alternatifleri"):
                    alt_df = pd.DataFrame(scores[:5], columns=["Oyuncu", "Skor", "Takım", "Lig", "Pozisyon"])
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Veri: FotMob · Metodoloji: Wyscout Index & Apunts (2024) · Geliştirici: M. Enes Şahin")
