import streamlit as st
import joblib
import torch
from sentence_transformers import util


# Load resources once
@st.cache_resource
def load_resources():
    media_df = joblib.load("media_df.pkl")
    embeddings = joblib.load("embeddings.pkl")
    return media_df, embeddings


media_df, embeddings = load_resources()


# Recommendation function
def recommend(title: str):
    match = media_df[media_df["title"].str.lower() == title.lower()]
    if match.empty:
        return []

    idx = match.index[0]
    query_embedding = embeddings[idx]
    cos_sim = util.cos_sim(query_embedding, embeddings)[0]
    top_k_scores, top_k_indices = torch.topk(cos_sim, k=100)

    results = []
    for score, i in zip(top_k_scores[1:], top_k_indices[1:]):
        media = media_df.loc[i.item()].to_dict()
        media["similarity"] = int(score * 100)
        if not media["rating"]:
            media["rating"] = "Not Yet Rated"
        results.append(media)
    return results


# Set page config
st.set_page_config(
    page_title="Next Drama - Because one drama is never enough",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://github.com/LakhindarPal/NextDrama",
        "Report a bug": "https://github.com/LakhindarPal/NextDrama/issues",
        "About": "NextDrama is a content-based recommendation app that helps you discover similar Asian dramas using title embeddings and rich metadata filters like genre, rating, and country and more.",
    },
)


@st.dialog("Details", width="large")
def details(media):
    st.header(media.get("title", "Unknown"), anchor=False)

    genres = media.get("genres", "")
    if genres:
        st.markdown(
            " ".join(
                f":blue-badge[{g.strip()}]" for g in genres.split(",") if g.strip()
            )
        )

    with st.expander("**Synopsis**"):
        st.text(media.get("synopsis", "---"))

    st.markdown(f"**Country:** {media.get('country', '—')}")
    date = media.get("date")
    if date and str(date) != "NaT":
        st.markdown(f"**Date:** {date.strftime('%Y-%m-%d')}")

    st.markdown(f"**Score:** {media.get('score', '—')}")
    st.markdown(f"**Rating:** {media.get('rating', '—')}")

    if media.get("type").lower() != "movie":
        st.markdown(f"**Episodes:** {int(media.get('episodes', '-'))}")
        st.markdown(f"**Network:** {media.get('network', '—')}")

    if media.get("directors"):
        st.markdown(f"**Directors:** {media['directors']}")
    if media.get("screenwriters"):
        st.markdown(f"**Screenwriters:** {media['screenwriters']}")

    cast = media.get("cast", {})
    if cast:
        st.markdown("**Cast:**")
        for role, persons in cast.items():
            if persons:
                st.markdown(f"- **{role.title()}**: {persons}")

    tags = media.get("tags", "")
    if tags:
        st.markdown(
            " ".join(
                f":violet-badge[{t.strip()}]" for t in tags.split(",") if t.strip()
            )
        )


# Sidebar: Filters
with st.sidebar:
    st.sidebar.header("Filters")

    types = st.sidebar.pills(
        "Format", sorted(media_df["type"].dropna().unique()), selection_mode="multi"
    )

    countries = st.sidebar.pills(
        "Origin", sorted(media_df["country"].dropna().unique()), selection_mode="multi"
    )

    score = st.sidebar.slider("Min Score", 0, 10, 5)

    min_year = media_df["date"].min().year
    max_year = media_df["date"].max().year
    st_year, end_year = st.sidebar.select_slider(
        "Release Year",
        options=list(range(min_year, max_year + 1)),
        value=(min_year, max_year),
    )

    rating_order = {
        "Any rating": 0,
        "G - All Ages": 1,
        "13+ - Teens 13 or older": 2,
        "15+ - Teens 15 or older": 3,
        "18+ Restricted (violence & profanity)": 4,
        "R - Restricted Screening (nudity & violence)": 5,
    }
    rating = st.sidebar.selectbox("Rating", list(rating_order.keys()))

    genres = st.sidebar.multiselect(
        "Genres",
        sorted(
            {
                g.strip()
                for genre_list in media_df["genres"].dropna()
                for g in genre_list.split(",")
            }
        ),
    )

    tags = st.sidebar.multiselect(
        "Tags",
        sorted(
            {
                t.strip()
                for tag_list in media_df["tags"].dropna()
                for t in tag_list.split(",")
            }
        ),
    )

st.title("🎬 NextDrama", anchor=False)
title = st.selectbox(
    "Enter a title",
    ["-- Select a title --"] + sorted(media_df["title"].dropna().unique()),
)

st.divider()

# Display recommendations
if title != "-- Select a title --":
    recommendations = recommend(title)

    # Filter function
    def passes_filters(media):
        if types and media.get("type") not in types:
            return False
        if countries and media.get("country") not in countries:
            return False
        try:
            if float(media.get("score", "0").split("/")[0]) < score:
                return False
        except:
            return False
        try:
            year = int(media["date"].strftime("%Y"))
            if not (st_year <= year <= end_year):
                return False
        except:
            return False
        if rating != "Any rating":
            if (
                media.get("rating")
                and rating_order[media.get("rating")] > rating_order[rating]
            ):
                return False
        if genres and not any(g in media.get("genres", "") for g in genres):
            return False
        if tags and not any(t in media.get("tags", "") for t in tags):
            return False
        return True

    # Apply filters
    filtered_recos = [m for m in recommendations if passes_filters(m)]

    # Show results
    if not filtered_recos:
        st.warning("No recommendations found.")

    for i in range(0, len(filtered_recos), 3):
        cols = st.columns(3, border=True)
        for j in range(3):
            if i + j < len(filtered_recos):
                media = filtered_recos[i + j]
                with cols[j]:
                    st.badge(
                        f"{media['similarity']}% Similar",
                        icon=":material/check:" if media["similarity"] > 65 else None,
                        color="blue" if media["similarity"] >= 65 else "gray",
                    )
                    st.image(
                        media.get("cover", "https://picsum.photos/300/422/?blur=2"),
                        use_container_width="always",
                    )
                    st.header(media["title"], anchor=False)
                    if st.button("Details", key=media["id"]):
                        details(media)
