import streamlit as st
import joblib
import torch
from sentence_transformers import util


# ---------- Load Data ----------
@st.cache_resource
def load_data():
    return joblib.load("media_df.pkl")


@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.pkl")


media_df = load_data()
embeddings = load_embeddings()


# ---------- Recommendation Function ----------
def recommend(title: str, top_k=20):
    match = media_df[media_df["title"].str.lower() == title.lower()]
    if match.empty:
        return []

    idx = match.index[0]
    query = embeddings[idx]
    cos_sim = util.cos_sim(query, embeddings)[0]
    top_k_scores, top_k_indices = torch.topk(cos_sim, k=top_k + 1)

    results = []
    for score, i in zip(top_k_scores[1:], top_k_indices[1:]):
        media = media_df.iloc[i.item()].to_dict()
        media["similarity"] = float(score)
        results.append(media)
    return results


# ---------- Custom Badge Renderer ----------
def render_badges(items, color="#eee", text_color="#000"):
    return " ".join(
        f"<span style='background:{color}; color:{text_color}; padding:2px 6px; border-radius:4px; margin-right:4px; font-size:0.8em'>{item.strip()}</span>"
        for item in items.split(",")
        if item.strip()
    )


# ---------- UI ----------
st.set_page_config("Drama Recommender", layout="wide")
st.title("🎬 Drama Recommendation System")

with st.sidebar:
    st.header("🔎 Filters")

    selected_title = st.selectbox(
        "🎥 Select a Title", sorted(media_df["title"].dropna().unique())
    )
    top_k = st.slider("📈 Number of Recommendations", 1, 20, 5)

    all_genres = sorted(
        {g.strip() for genres in media_df["genres"].dropna() for g in genres.split(",")}
    )
    all_tags = sorted(
        {t.strip() for tags in media_df["tags"].dropna() for t in tags.split(",")}
    )
    all_types = sorted(media_df["type"].dropna().unique())
    all_countries = sorted(media_df["country"].dropna().unique())

    selected_genres = st.multiselect("🎭 Genre", all_genres)
    selected_tags = st.multiselect("🏷️ Tags", all_tags)
    selected_type = st.selectbox("🎬 Type", ["Any"] + all_types)
    selected_country = st.selectbox("🌍 Country", ["Any"] + all_countries)

# ---------- Recommend + Filter ----------
if st.button("🚀 Recommend"):
    recs = recommend(selected_title, top_k=top_k)

    def passes_filters(media):
        if selected_genres and not any(
            g in media.get("genres", "") for g in selected_genres
        ):
            return False
        if selected_tags and not any(t in media.get("tags", "") for t in selected_tags):
            return False
        if selected_type != "Any" and media.get("type", "") != selected_type:
            return False
        if selected_country != "Any" and media.get("country", "") != selected_country:
            return False
        return True

    filtered_recs = [m for m in recs if passes_filters(m)]

    if not filtered_recs:
        st.warning("No recommendations found with current filters.")
    else:
        for media in filtered_recs:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(
                    media.get(
                        "cover", "https://via.placeholder.com/120x180.png?text=No+Image"
                    ),
                    width=140,
                )

            with col2:
                st.markdown(
                    f"### {media['title']}  \n**🔍 Similarity:** `{media['similarity']:.4f}`"
                )

                st.markdown(f"**📺 Type:** {media.get('type', '-')}")
                st.markdown(f"**🎞️ Episodes:** {media.get('episodes', '-')}")
                st.markdown(f"**📡 Network:** {media.get('network', '-')}")
                st.markdown(f"**🌍 Country:** {media.get('country', '-')}")
                st.markdown(f"**⭐ Score:** {media.get('score', '-')}")
                st.markdown(f"**📅 Date:** {media.get('date', '-')}")

                if media.get("genres"):
                    st.markdown("**🎭 Genres:**", unsafe_allow_html=True)
                    st.markdown(
                        render_badges(media["genres"], "#f0f0f0", "#000"),
                        unsafe_allow_html=True,
                    )

                if media.get("tags"):
                    st.markdown("**🏷️ Tags:**", unsafe_allow_html=True)
                    st.markdown(
                        render_badges(media["tags"], "#dfe7fd", "#1d3557"),
                        unsafe_allow_html=True,
                    )

                with st.expander("📖 Synopsis / Details"):
                    st.write(f"**📃 Synopsis:** {media.get('synopsis', '-')}")
                    st.write(f"**🎬 Directors:** {media.get('directors', '-')}")
                    st.write(f"**✍️ Screenwriters:** {media.get('screenwriters', '-')}")
                    st.write(f"**🎭 Cast:** {media.get('cast', '-')}")
                    st.write(f"**🔞 Rating:** {media.get('rating', '-')}")

            st.markdown("---")
