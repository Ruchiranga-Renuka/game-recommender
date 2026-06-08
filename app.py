import pickle
import textwrap
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set page config
st.set_page_config(
    page_title="Game Recommender",
    page_icon="🎮",
    layout="wide"
)

# Paths
DATA_DIR = Path(__file__).resolve().parent
GAMES_PKL = DATA_DIR / "games.pkl"
SIMILARITY_PKL = DATA_DIR / "similarity.pkl"
GAMES_CSV = DATA_DIR / "games.csv"

# Stylings
STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
                    linear-gradient(135deg, #030712 0%, #0b1528 100%);
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        line-height: 1.1;
        margin-bottom: 0.25rem;
        font-family: 'Outfit', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .card {
        border-radius: 20px;
        padding: 1.5rem;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    .hero-card {
        display: flex;
        gap: 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
        align-items: center;
    }
    @media (max-width: 992px) {
        .hero-card {
            flex-direction: column;
            gap: 1.5rem;
            padding: 1.5rem;
        }
        .hero-image-container {
            width: 100% !important;
        }
    }
    .hero-image-container {
        width: 35%;
        flex-shrink: 0;
    }
    .hero-image {
        width: 100%;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        object-fit: cover;
        aspect-ratio: 16/9;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-details {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        flex-grow: 1;
        width: 100%;
    }
    .hero-label {
        font-size: 0.75rem;
        font-weight: 800;
        color: #818cf8;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
        display: block;
    }
    .hero-title-text {
        font-size: 2.25rem;
        font-weight: 800;
        color: #f9fafb;
        margin: 0 0 0.5rem 0;
        font-family: 'Outfit', sans-serif;
        line-height: 1.2;
    }
    .hero-developer {
        font-size: 0.95rem;
        color: #9ca3af;
        margin: 0 0 1rem 0;
    }
    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 1.25rem;
    }
    .stat-item {
        display: flex;
        flex-direction: column;
        padding-right: 1.5rem;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .steam-button {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: #ffffff !important;
        padding: 0.6rem 1.25rem;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        width: fit-content;
    }
    .steam-button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        transform: translateY(-2px);
        text-decoration: none;
    }
    
    .card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        height: 100%;
    }
    .recommendation-card {
        border-radius: 16px;
        padding: 1.25rem;
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 15px 30px rgba(96, 165, 250, 0.15);
        background: rgba(17, 24, 39, 0.75);
    }
    .image-container {
        position: relative;
        width: 100%;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .game-image {
        width: 100%;
        transition: transform 0.5s ease;
        display: block;
        object-fit: cover;
        aspect-ratio: 16/9;
    }
    .recommendation-card:hover .game-image {
        transform: scale(1.08);
    }
    .recommendation-title {
        margin: 0 0 0.5rem 0;
        color: #f9fafb;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.3;
        font-family: 'Outfit', sans-serif;
    }
    .game-meta {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .price-tag {
        font-size: 0.9rem;
        font-weight: 700;
        color: #34d399;
    }
    .rating-tag {
        font-size: 0.8rem;
        font-weight: 600;
        color: #fbbf24;
    }
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 0.5rem;
    }
    .sidebar .sidebar-content {
        background: rgba(17, 24, 39, 0.85);
        border-radius: 20px;
        padding: 1.25rem;
    }
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 1rem;
    }
    .warning-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.3);
        margin-bottom: 1rem;
    }
</style>
"""

@st.cache_data(show_spinner="Loading games catalog...")
def load_games():
    if GAMES_CSV.exists():
        try:
            return pd.read_csv(GAMES_CSV)
        except Exception:
            pass

    if GAMES_PKL.exists():
        try:
            return pickle.load(GAMES_PKL.open('rb'))
        except Exception:
            pass

    return None

@st.cache_resource(show_spinner="Loading similarity matrix (5.8 GB)...")
def load_similarity(force=False):
    if not force:
        return "BYPASSED"
    if not SIMILARITY_PKL.exists():
        return None

    try:
        # Load the large matrix
        return pickle.load(SIMILARITY_PKL.open('rb'))
    except MemoryError:
        return "MEMORY_ERROR"
    except Exception as exc:
        return exc

@st.cache_data(show_spinner="Building content matrix...")
def build_content_matrix(games):
    # Support 'tags' column for games.pkl compatibility
    text_columns = [
        col for col in ["genres", "steamspy_tags", "developer", "publisher", "categories", "tags"] if col in games.columns
    ]
    if not text_columns:
        return None

    games = games.copy()
    games["content_features"] = games[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    return vectorizer.fit_transform(games["content_features"])

def get_similarity_recommendations(selected_game, games, similarity):
    # Find game index
    game_index = games[games["name"] == selected_game].index[0]
    pos_idx = games.index.get_loc(game_index)
    
    distances = similarity[pos_idx]
    games_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    # Return game names instead of indices
    return [games["name"].iloc[idx] for idx, _ in games_list]

def get_content_recommendations(selected_game, games, content_matrix):
    game_index = games[games["name"] == selected_game].index[0]
    pos_idx = games.index.get_loc(game_index)
    
    similarities = linear_kernel(content_matrix[pos_idx], content_matrix).flatten()
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [idx for idx in similar_indices if idx != pos_idx]
    # Return game names instead of indices
    return [games["name"].iloc[idx] for idx in similar_indices[:5]]

def render_header():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🎮 Game Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Interactive Steam game discovery system powered by smart semantic matches.</div>',
        unsafe_allow_html=True,
    )

def render_sidebar(game_count: int, fallback: bool, bypassed: bool):
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total games in database:** {game_count}")
    
    if fallback:
        if bypassed:
            st.sidebar.markdown('<div class="info-badge">💡 Using on-the-fly TF-IDF</div>', unsafe_allow_html=True)
            st.sidebar.markdown("**Mode:** TF-IDF Content-Based fallback (optimized for fast startup and low RAM)")
        else:
            st.sidebar.markdown('<div class="warning-badge">⚠️ Error Loading Matrix</div>', unsafe_allow_html=True)
            st.sidebar.markdown("**Mode:** Content-based fallback (Matrix too large or failed to load)")
    else:
        st.sidebar.markdown('<div class="info-badge">🚀 Precomputed Matrix Active</div>', unsafe_allow_html=True)
        st.sidebar.markdown("**Mode:** Precomputed Similarity Matrix")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Technical Info")
    st.sidebar.markdown(
        "TF-IDF recommendation matches genres, tags, developer, and publisher metadata on-the-fly. "
        "It is extremely lightweight and fast."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with Streamlit & Scikit-Learn.")

def get_platform_icons(platforms_str):
    if not isinstance(platforms_str, str):
        return ""
    platforms = platforms_str.lower().split(";")
    icons = []
    if "windows" in platforms:
        icons.append("🪟")
    if "mac" in platforms:
        icons.append("🍎")
    if "linux" in platforms:
        icons.append("🐧")
    return " ".join(icons)

def format_price(price):
    if pd.isna(price):
        return "N/A"
    try:
        price_val = float(price)
        return "Free" if price_val == 0 else f"${price_val:.2f}"
    except (ValueError, TypeError):
        price_str = str(price).strip().lower()
        if price_str in ["0", "0.0", "free"]:
            return "Free"
        return str(price)

def build_placeholder_svg(width=600, height=337, text="No Image"):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="#0f172a"/>'
        f'<text x="50%" y="50%" fill="#cbd5e1" font-family="Inter, sans-serif" font-size="28" text-anchor="middle" alignment-baseline="middle">{text}</text>'
        '</svg>'
    )
    return "data:image/svg+xml;charset=UTF-8," + urllib.parse.quote(svg, safe='')

def get_game_image_sources(game_data):
    image_url = game_data.get("image_url", "")
    if isinstance(image_url, str) and image_url.strip():
        return [image_url.strip()]

    appid = game_data.get("appid")
    if pd.notna(appid):
        try:
            appid_int = int(appid)
        except (ValueError, TypeError):
            return []
        return [
            f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid_int}/header.jpg",
            f"https://steamcdn-a.akamaihd.net/steam/apps/{appid_int}/header.jpg",
        ]

    return []

def get_image_url(game_data):
    image_sources = get_game_image_sources(game_data)
    if image_sources:
        return image_sources[0]
    return "https://placehold.co/600x337/1e293b/cbd5e1?text=No+Image"


def render_hero_card(game_data):
    title = game_data["name"]
    developer = game_data.get("developer", "Unknown Developer")
    publisher = game_data.get("publisher", "Unknown Publisher")
    release_date = game_data.get("release_date", "Unknown Release Date")
    price = format_price(game_data.get("price", "N/A"))

    pos = game_data.get("positive_ratings", 0)
    neg = game_data.get("negative_ratings", 0)
    total = pos + neg
    if total > 0:
        rating_pct = int((pos / total) * 100)
        rating_str = f"👍 {rating_pct}% ({total:,} reviews)"
    else:
        rating_str = "No reviews"

    platforms_str = game_data.get("platforms", "")
    platform_icons = get_platform_icons(platforms_str)

    genres = game_data.get("genres", "")
    if not genres:
        genres = game_data.get("tags", "")
    genre_list = [g.strip() for g in str(genres).replace(";", ",").split(",") if g.strip()][:4]
    badges = " ".join([f"`{g}`" for g in genre_list])

    image_url = get_image_url(game_data)
    cols = st.columns([1.3, 2.7], gap="large")
    with cols[0]:
        st.image(image_url, width=320)
    with cols[1]:
        st.markdown(f"### {title}", unsafe_allow_html=True)
        st.markdown(f"**Developer:** {developer}  \n**Publisher:** {publisher}")
        st.markdown(f"**Release Date:** {release_date}  \n**Price:** {price}  \n**Platforms:** {platform_icons}")
        st.markdown(f"**Rating:** {rating_str}")
        if badges:
            st.markdown(f"**Genres:** {badges}")
        appid = game_data.get("appid")
        if pd.notna(appid):
            steam_link = f"https://store.steampowered.com/app/{int(appid)}/"
            st.markdown(f"[View on Steam Store ↗]({steam_link})")


def render_game_card(game_data, col):
    title = game_data["name"]
    developer = game_data.get("developer", "Unknown Dev")
    price = format_price(game_data.get("price", "N/A"))

    pos = game_data.get("positive_ratings", 0)
    neg = game_data.get("negative_ratings", 0)
    total = pos + neg
    if total > 0:
        rating_pct = int((pos / total) * 100)
        rating_str = f"👍 {rating_pct}%"
    else:
        rating_str = "No ratings"

    platforms_str = game_data.get("platforms", "")
    platform_icons = get_platform_icons(platforms_str)

    genres = game_data.get("genres", "")
    if not genres:
        genres = game_data.get("tags", "")
    genre_list = [g.strip() for g in str(genres).replace(";", ",").split(",") if g.strip()][:2]
    badges = " ".join([f"`{g}`" for g in genre_list])

    image_url = get_image_url(game_data)
    col.image(image_url, width=280)
    col.markdown(f"### {title}")
    col.markdown(f"*By {developer}*")
    if badges:
        col.markdown(badges)
    col.markdown(f"**{price}**  \n{platform_icons}  \n{rating_str}")

    appid = game_data.get("appid")
    if pd.notna(appid):
        steam_link = f"https://store.steampowered.com/app/{int(appid)}/"
        col.markdown(f"[View on Steam ↗]({steam_link})")


def render_recommendations(recommendations, games):
    st.markdown('<h3 style="margin-top: 1rem; color: #f9fafb; font-family: \"Outfit\", sans-serif;">Top 5 Recommended Matches</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 1.5rem;">Click on any card to view the game directly on the Steam Store.</p>', unsafe_allow_html=True)
    cols = st.columns(5, gap="medium")
    for col, title in zip(cols, recommendations):
        game_rows = games[games["name"] == title]
        if not game_rows.empty:
            game_data = game_rows.iloc[0]
            render_game_card(game_data, col)
        else:
            col.markdown(f"### {title}")
def main():
    render_header()

    games = load_games()
    if games is None:
        st.error("Unable to load games data. Make sure games.pkl or games.csv exists in the app folder.")
        return

    if "name" not in games.columns:
        st.error("The loaded games dataset does not contain a 'name' column.")
        return

    # Option in sidebar to load the huge similarity matrix
    st.sidebar.title("Options")
    load_precomputed = st.sidebar.checkbox(
        "Load Precomputed Matrix (5.8 GB)",
        value=False,
        help="WARNING: Loading the precomputed similarity matrix requires ~6GB of RAM and may crash the app if memory is insufficient."
    )

    similarity = load_similarity(force=load_precomputed)
    bypassed_similarity = isinstance(similarity, str) and similarity == "BYPASSED"
    is_memory_error = isinstance(similarity, str) and similarity == "MEMORY_ERROR"
    is_exception = isinstance(similarity, Exception)
    is_none = similarity is None
    fallback_mode = bypassed_similarity or is_memory_error or is_exception or is_none
    content_matrix = None

    if fallback_mode:
        content_matrix = build_content_matrix(games)
        if content_matrix is None:
            st.error("Unable to build content-based recommendations because metadata fields are missing.")
            return

    render_sidebar(len(games), fallback_mode, bypassed_similarity)

    with st.container():
        # Clean two-column header selection
        col_select, col_info = st.columns([1.8, 1.2], gap="large")

        with col_select:
            st.markdown('<div class="card" style="padding: 1.25rem; margin-bottom: 0;">', unsafe_allow_html=True)
            selected_game = st.selectbox(
                "🎯 Choose a game to find similar recommendations:",
                games["name"].values,
                index=0,
                help="Type to search or scroll to choose.",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="card" style="padding: 1.25rem; margin-bottom: 0; font-size: 0.9rem; height: 100%; display: flex; align-items: center;">', unsafe_allow_html=True)
            if fallback_mode:
                st.markdown(
                    "⚡ **Status:** Running on **TF-IDF Content Matching**. "
                    "Recommendations update instantly as you change the selection."
                )
            else:
                st.markdown(
                    "🚀 **Status:** Running on **Precomputed Similarity Matrix**. "
                    "Recommendations update instantly as you change the selection."
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # Selected Game Details Preview
    selected_game_rows = games[games["name"] == selected_game]
    if not selected_game_rows.empty:
        st.markdown('<h3 style="color: #f9fafb; font-family: \'Outfit\', sans-serif; margin-top: 2rem; margin-bottom: 0.75rem;">Selected Game</h3>', unsafe_allow_html=True)
        selected_game_data = selected_game_rows.iloc[0]
        render_hero_card(selected_game_data)

    # Compute and render recommendations automatically
    if selected_game:
        try:
            if fallback_mode:
                recs = get_content_recommendations(selected_game, games, content_matrix)
            else:
                recs = get_similarity_recommendations(selected_game, games, similarity)
            
            if recs:
                st.markdown("---")
                render_recommendations(recs, games)
        except Exception as exc:
            st.error(f"Recommendation error: {exc}")

if __name__ == "__main__":
    main()
