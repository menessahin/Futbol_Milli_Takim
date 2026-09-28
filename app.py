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

# ── CSS ──────────────────────────────────────────────────────────────────────
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
    display: flex; justify-content: space-between; align-items: center;
}
.pos-card:hover { border-color: #e63946; }
.pos-card-title { font-family: 'Barlow Condensed', sans-serif; font-size: 1.1rem; font-weight: 600; color: #e63946; margin-bottom: 0.25rem; }
.player-name { font-size: 1.3rem; font-weight: 600; color: #ffffff; }
.player-meta { font-size: 0.75rem; color: #6b7fa3; margin-top: 0.2rem; }
.score-badge {
    background: #e63946; color: white;
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem; font-weight: 700;
    padding: 0.3rem 0.8rem; border-radius: 4px;
}
.stSlider > div > div > div { background: #e63946 !important; }
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #6b7fa3; }
.stTabs [aria-selected="true"] { color: #e63946 !important; }
.streamlit-expanderHeader { background: #111827 !important; color: #e8eaf0 !important; border-radius: 6px; }
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

# ── SKOR METODOLOJİSİ ────────────────────────────────────────────────────────
# pct_rank sütunları 0-100 arasında ligteki yüzdelik dilimi verir.
# Bu değerleri direkt kullanıyoruz (zaten normalize edilmiş).
# Ek olarak main_rating'i form skoru olarak ekliyoruz (%15 ağırlıkla).
# goals_prevented negatif olabileceği için özel sign-aware normalizasyon uygulanır.

def get_pct(df, col, player_name):
    """pct_rank sütunundan oyuncunun değerini al (0-100). Yoksa 50 ver."""
    pct_col = f"{col}__pct_rank"
    row = df[df['name'] == player_name]
    if row.empty or pct_col not in df.columns:
        return 50.0
    val = row[pct_col].values[0]
    return float(val) if pd.notna(val) else 50.0

def get_form_score(df, player_name):
    """main_rating'i 0-100 skalasına çevir. FotMob rating 5-9 arası."""
    row = df[df['name'] == player_name]
    if row.empty:
        return 50.0
    rating = row['main_rating'].values[0]
    if pd.isna(rating):
        return 50.0
    # FotMob: 5.0 (min) → 9.0 (max) → 0-100 skalası
    normalized = (float(rating) - 5.0) / (9.0 - 5.0) * 100.0
    return max(0.0, min(100.0, normalized))

def get_goals_prevented_score(df, player_name):
    """
    goals_prevented negatif olabilir (kötü kaleci → gol izni verir).
    Tüm takımlar içinde min-max normalize et, sonra 0-100'e çevir.
    """
    col = 'goalkeeping__goals_prevented'
    if col not in df.columns:
        return 50.0
    series = df[col].dropna()
    if series.empty:
        return 50.0
    mn, mx = series.min(), series.max()
    row = df[df['name'] == player_name]
    if row.empty:
        return 50.0
    val = row[col].values[0]
    if pd.isna(val):
        return 50.0
    if mx == mn:
        return 50.0
    return float((val - mn) / (mx - mn) * 100.0)

def compute_composite_score(player_name, metrics_weights, form_weight=0.15):
    """
    Her metrik için pct_rank kullanarak ağırlıklı kompozit skor hesapla.
    form_weight: main_rating'in toplam skordaki payı.
    Kalan (1-form_weight) metrik ağırlıklarına dağıtılır.
    """
    form_score = get_form_score(df, player_name)

    metric_score = 0.0
    total_metric_w = sum(w for _, w in metrics_weights.values())

    for col, (label, w) in metrics_weights.items():
        if total_metric_w == 0:
            normalized_w = 0
        else:
            normalized_w = w / total_metric_w

        if col == 'goalkeeping__goals_prevented':
            pct = get_goals_prevented_score(df, player_name)
        else:
            pct = get_pct(df, col, player_name)

        metric_score += pct * normalized_w

    # Form ağırlığı karıştır
    final = metric_score * (1 - form_weight) + form_score * form_weight
    return round(final, 1)

# ── FORMASYON & POZİSYON TANIMI ──────────────────────────────────────────────
POSITIONS = {
    "Kaleci": {
        "abbr": "KAL",
        "candidates": ["Altay Bayındır", "Muhammed Şengezer", "Okan Kocuk", "Uğurcan Çakir"],
        "metrics": {
            # Metrik: (Türkçe etiket, ağırlık)
            # Kaynak: Apunts (2024) kaleci indeksi + Wyscout GK modeli
            "goalkeeping__save_percentage":    ("Kurtarış %",        0.30),
            "goalkeeping__goals_prevented":    ("Önlenen Gol",       0.25),
            "goalkeeping__clean_sheets":       ("Gol Yememe",        0.18),
            "goalkeeping__high_claims":        ("Hava Topu",         0.12),
            "distribution__pass_accuracy":     ("Pas İsabeti",       0.15),
        }
    },
    "Sağ Stoper": {
        "abbr": "RST",
        "candidates": ["Merih Demiral", "Ozan Kabak", "Samet Akaydin"],
        "metrics": {
            "defending__interceptions":              ("Top Kapma",        0.20),
            "defending__tackles":                    ("Müdahale",         0.18),
            "possession__aerials_won_pct":           ("Hava Topu %",      0.18),
            "defending__clearances":                 ("Uzaklaştırma",     0.14),
            "defending__recoveries":                 ("Top Kazanma",      0.12),
            "defending__blocked_scoring_attempt":    ("Şut Blok",         0.08),
            "passing__pass_accuracy":                ("Pas İsabeti",      0.10),
        }
    },
    "Sol Stoper": {
        "abbr": "LST",
        "candidates": ["Emirhan Topçu", "Abdülkerim Bardakci", "Adil Demirbağ"],
        "metrics": {
            "defending__interceptions":              ("Top Kapma",        0.20),
            "defending__tackles":                    ("Müdahale",         0.18),
            "possession__aerials_won_pct":           ("Hava Topu %",      0.18),
            "defending__clearances":                 ("Uzaklaştırma",     0.14),
            "defending__recoveries":                 ("Top Kazanma",      0.12),
            "defending__blocked_scoring_attempt":    ("Şut Blok",         0.08),
            "passing__pass_accuracy":                ("Pas İsabeti",      0.10),
        }
    },
    "Sağ Bek": {
        "abbr": "SBK",
        "candidates": ["Zeki Çelik"],
        "metrics": {
            "defending__tackles":              ("Müdahale",             0.25),
            "defending__interceptions":        ("Top Kapma",            0.22),
            "passing__xa":                     ("Beklenen Asist",       0.20),
            "passing__cross_accuracy":         ("Orta İsabeti",         0.18),
            "possession__duels_won_pct":       ("İkili Kazanma %",      0.15),
        }
    },
    "Sol Bek": {
        "abbr": "LBK",
        "candidates": ["Eren Elmalı", "Ferdi Kadıoğlu"],
        "metrics": {
            "passing__xa":                     ("Beklenen Asist",       0.28),
            "passing__cross_accuracy":         ("Orta İsabeti",         0.18),
            "defending__tackles":              ("Müdahale",             0.20),
            "defending__interceptions":        ("Top Kapma",            0.18),
            "possession__duels_won_pct":       ("İkili Kazanma %",      0.16),
        }
    },
    "6 Numara": {
        "abbr": "6",
        "candidates": ["Ismail Yüksek", "Melih Kabasakal", "Salih Özcan"],
        "metrics": {
            "defending__interceptions":              ("Top Kapma",        0.28),
            "defending__tackles":                    ("Müdahale",         0.25),
            "passing__pass_accuracy":                ("Pas İsabeti",      0.22),
            "defending__possession_won_final_3rd":   ("Son Üçte Kazanma", 0.15),
            "possession__duels_won_pct":             ("İkili Kazanma %",  0.10),
        }
    },
    "8 Numara": {
        "abbr": "8",
        "candidates": ["Orkun Kökcü", "Demir Tıknaz", "Bartuğ Elmaz"],
        "metrics": {
            "passing__xa":                               ("Beklenen Asist",      0.22),
            "passing__chances_created":                  ("Fırsat Yaratma",      0.20),
            "passing__line-breaking_passes":             ("Hat Kıran Pas",       0.18),
            "shooting__xg":                              ("Beklenen Gol",        0.18),
            "possession__touches_in_opposition_box":     ("Rakip Ceza Dokunuş",  0.12),
            "defending__tackles":                        ("Müdahale",            0.10),
        }
    },
    "Sağ Açık": {
        "abbr": "RW",
        "candidates": ["Yunus Akgün", "Oğuz Aydın", "İrfan Kahveci"],
        "metrics": {
            "shooting__xg":                              ("Beklenen Gol",        0.25),
            "passing__xa":                               ("Beklenen Asist",      0.20),
            "possession__dribbles_success_rate":         ("Dribling %",          0.20),
            "possession__touches_in_opposition_box":     ("Rakip Ceza Dokunuş",  0.18),
            "shooting__shots_on_target":                 ("İsabetli Şut",        0.17),
        }
    },
    "10 Numara": {
        "abbr": "10",
        "candidates": ["Arda Güler", "Can Uzun"],
        "metrics": {
            "passing__xa":                               ("Beklenen Asist",      0.28),
            "passing__chances_created":                  ("Fırsat Yaratma",      0.25),
            "passing__big_chances_created":              ("Büyük Fırsat",        0.18),
            "shooting__xg":                              ("Beklenen Gol",        0.17),
            "possession__touches_in_opposition_box":     ("Rakip Ceza Dokunuş",  0.12),
        }
    },
    "Sol Açık": {
        "abbr": "LW",
        "candidates": ["Aral Şimşir", "İlhan Fakılı"],
        "metrics": {
            "shooting__xg":                              ("Beklenen Gol",        0.25),
            "passing__xa":                               ("Beklenen Asist",      0.22),
            "possession__dribbles_success_rate":         ("Dribling %",          0.20),
            "possession__touches_in_opposition_box":     ("Rakip Ceza Dokunuş",  0.18),
            "shooting__shots_on_target":                 ("İsabetli Şut",        0.15),
        }
    },
    "Forvet": {
        "abbr": "FW",
        "candidates": ["Kerem Aktürkoglu", "Barış Alper Yılmaz", "Deniz Gül"],
        "metrics": {
            "shooting__xg":                              ("Beklenen Gol",        0.28),
            "shooting__goals":                           ("Gol",                 0.25),
            "shooting__xgot":                            ("Şut Kalitesi",        0.20),
            "possession__touches_in_opposition_box":     ("Rakip Ceza Dokunuş",  0.15),
            "possession__aerials_won_pct":               ("Hava Topu %",         0.12),
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

# ── EN İYİ OYUNCU SEÇİMİ ─────────────────────────────────────────────────────
def get_best_player(pos_name, pos_config, weights_override, used_names):
    candidates = [n for n in pos_config['candidates'] if n not in used_names]
    if not candidates:
        return None, []

    metrics = {}
    for col, (label, default_w) in pos_config['metrics'].items():
        metrics[col] = (label, weights_override.get(col, default_w))

    scores = []
    for name in candidates:
        row = df[df['name'] == name]
        if row.empty:
            continue
        s = compute_composite_score(name, metrics)
        r = row.iloc[0]
        scores.append((name, s, r.get('team', ''), r.get('league', ''), r.get('position_primary', '')))

    scores.sort(key=lambda x: x[1], reverse=True)
    return (scores[0] if scores else None), scores

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

st.markdown("""
Oyuncu skorları **lig yüzdelik dilimlerine (percentile rank)** ve **son sezon form puanına** göre hesaplanır.
Her mevki için metrik ağırlıklarını ayarlayabilirsin.
""")

# ── ANA LAYOUT ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-title">Mevki Ayarları</div>', unsafe_allow_html=True)

    lineup = {}
    used_names = set()
    all_scores = {}

    for pos_name in FORMATION_ORDER:
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
                    min_value=0, max_value=100, value=pct, step=5,
                    key=f"{pos_name}_{col}"
                )
                weights_override[col] = new_val / 100.0

            if st.button(f"Bu mevki için hesapla", key=f"calc_{pos_name}", type="primary"):
                st.session_state.weights[pos_name] = weights_override

        current_weights = st.session_state.weights.get(pos_name, {})
        best, scores = get_best_player(pos_name, pos_config, current_weights, used_names)

        if best:
            lineup[pos_name] = best
            used_names.add(best[0])
            all_scores[pos_name] = scores

with right_col:
    st.markdown('<div class="section-title">Önerilen 11</div>', unsafe_allow_html=True)

    if lineup:
        # ── SVG PITCH ─────────────────────────────────────────────────────────
        W, H = 400, 580

        # Koordinatlar: X → saha sol=küçük, sağ=büyük
        # Futbol konvansiyonu: Sağ Bek = sahada sağ = SVG'de büyük X
        formation_slots = {
            "Kaleci":     (200, 520),
            "Sol Bek":    ( 60, 400),   # SVG solda = futbolda sol
            "Sol Stoper": (150, 400),
            "Sağ Stoper": (250, 400),
            "Sağ Bek":    (340, 400),   # SVG sağda = futbolda sağ
            "6 Numara":   (140, 290),
            "8 Numara":   (260, 290),
            "Sol Açık":   ( 60, 170),   # SVG solda = futbolda sol kanat
            "10 Numara":  (200, 170),
            "Sağ Açık":   (340, 170),   # SVG sağda = futbolda sağ kanat
            "Forvet":     (200,  55),
        }

        def player_card_svg(cx, cy, pos_abbr, player_name, score):
            short = player_name.split()[-1] if player_name else "—"
            cw, ch = 76, 40
            x0 = cx - cw / 2
            y0 = cy - ch / 2
            return f"""
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{cw}" height="{ch}"
        rx="6" fill="rgba(8,13,26,0.85)" stroke="#e63946" stroke-width="1.4"/>
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{cw}" height="14"
        rx="6" fill="#e63946"/>
  <rect x="{x0:.1f}" y="{y0+8:.1f}" width="{cw}" height="6" fill="#e63946"/>
  <text x="{cx:.1f}" y="{y0+10.5:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Barlow Condensed,sans-serif" font-size="8" font-weight="700"
        fill="#ffffff" letter-spacing="0.5">{pos_abbr}</text>
  <text x="{cx:.1f}" y="{y0+27:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Inter,sans-serif" font-size="9.5" font-weight="600"
        fill="#ffffff">{short}</text>"""

        pitch_lines = f"""
  <rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="url(#grass)"/>
  <rect x="20" y="15" width="{W-40}" height="{H-30}" rx="4"
        fill="none" stroke="#2d6a2d" stroke-width="1.5"/>
  <line x1="20" y1="{H//2}" x2="{W-20}" y2="{H//2}"
        stroke="#2d6a2d" stroke-width="1.2"/>
  <circle cx="{W//2}" cy="{H//2}" r="38"
          fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <circle cx="{W//2}" cy="{H//2}" r="2.5" fill="#2d6a2d"/>
  <rect x="105" y="15" width="190" height="72"
        fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <rect x="148" y="15" width="104" height="30"
        fill="none" stroke="#2d6a2d" stroke-width="1.0"/>
  <rect x="105" y="{H-87}" width="190" height="72"
        fill="none" stroke="#2d6a2d" stroke-width="1.2"/>
  <rect x="148" y="{H-45}" width="104" height="30"
        fill="none" stroke="#2d6a2d" stroke-width="1.0"/>
  <circle cx="{W//2}" cy="56" r="2" fill="#2d6a2d"/>
  <circle cx="{W//2}" cy="{H-56}" r="2" fill="#2d6a2d"/>
  <!-- Filigran -->
  <text x="{W-14}" y="22" text-anchor="end" dominant-baseline="middle"
        font-family="Barlow Condensed,sans-serif" font-size="11" font-weight="700"
        fill="rgba(255,255,255,0.45)" letter-spacing="0.8">M.Enes SAHIN</text>
  <text x="{W-14}" y="35" text-anchor="end" dominant-baseline="middle"
        font-family="Inter,sans-serif" font-size="8" font-weight="400"
        fill="rgba(255,255,255,0.30)" letter-spacing="0.5">Data Scientist</text>"""

        cards_svg = ""
        for pos_name, (cx, cy) in formation_slots.items():
            player = lineup.get(pos_name)
            abbr = POSITIONS[pos_name]["abbr"]
            if player:
                cards_svg += player_card_svg(cx, cy, abbr, player[0], player[1])
            else:
                # Boş slot
                cw, ch = 76, 52
                x0 = cx - cw/2; y0 = cy - ch/2
                cards_svg += f"""
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{cw}" height="{ch}"
        rx="6" fill="rgba(8,13,26,0.4)" stroke="#2d6a2d" stroke-width="1" stroke-dasharray="4"/>
  <text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-family="Barlow Condensed,sans-serif" font-size="9" fill="#4a7a4a">{abbr}</text>"""

        svg_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body{{margin:0;padding:0;background:transparent;}}
svg{{width:100%;max-width:420px;display:block;margin:0 auto;border-radius:10px;}}</style>
</head><body>
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

        components.html(svg_full, height=H + 20, scrolling=False)

        # ── KADRO LİSTESİ ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Oyuncu Detayları</div>', unsafe_allow_html=True)

        for pos_name in FORMATION_ORDER:
            player = lineup.get(pos_name)
            if not player:
                continue
            name, score, team, league, primary_pos = player
            form = get_form_score(df, name)
            st.markdown(f"""
            <div class="pos-card">
                <div>
                    <div class="pos-card-title">{pos_name}</div>
                    <div class="player-name">{name}</div>
                    <div class="player-meta">{team} · {league} · Form: {form:.0f}/100</div>
                </div>
                <div class="score-badge">{score}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── ALTERNATİFLER ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Alternatif Oyuncular</div>', unsafe_allow_html=True)
        for pos_name, scores in all_scores.items():
            if scores and len(scores) > 1:
                with st.expander(f"{pos_name} alternatifleri"):
                    alt_df = pd.DataFrame(
                        scores[:5],
                        columns=["Oyuncu", "Skor", "Takım", "Lig", "Pozisyon"]
                    )
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)

# ── METODOLOJİ NOTU ──────────────────────────────────────────────────────────
with st.expander("📊 Metodoloji"):
    st.markdown("""
    **Skor Hesaplama Yöntemi**

    Her oyuncu için kompozit skor iki bileşenden oluşur:

    - **Metrik Skoru (%85):** Her metrik için ligteki yüzdelik dilim (percentile rank) kullanılır.
      Yüzdelik dilim, oyuncunun o metrikte ligdeki oyuncuların kaçını geride bıraktığını gösterir.
      Ağırlıklar literatür referanslıdır: *Apunts Journal of Physical Education (2024)* ve *Wyscout Index*.

    - **Form Skoru (%15):** FotMob sezon rating'i (5.0–9.0 skalası) 0-100'e çevrilerek eklenir.

    **Negatif Metrikler:** `goals_prevented` negatif olabileceğinden (kötü kaleci pozitif gol izni verir),
    bu metrik için tüm kadroda min-max normalizasyon uygulanır.

    **Pozisyon Koordinatları:** Sahada sağ = SVG'de büyük X değeri (standart futbol görünümü).
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Veri: FotMob · Metodoloji: Wyscout Index & Apunts (2024) · Geliştirici: M. Enes Şahin · menessahin.github.io")
