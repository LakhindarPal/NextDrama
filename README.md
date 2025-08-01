# 🎬 NextEpisode

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nextepisode.streamlit.app)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NextEpisode** is a content-based recommendation system that helps users find similar Asian dramas, movies, specials, and TV shows based on deep semantic similarity using Sentence Transformers.

> 🌐 **Live Demo:** [nextepisode.streamlit.app](https://nextepisode.streamlit.app)

---

## 🔍 Features

- ✅ Deep semantic matching using `all-mpnet-base-v2` from Sentence Transformers
- ✅ Smart content-based recommendations (no user history required)
- ✅ Filter by genre, country, type, tags, and release year
- ✅ Clean, responsive UI with dynamic card layouts
- ✅ Clickable cards reveal full metadata and synopsis
- ✅ Optimized with cosine similarity for 20,000+ items

---

## 📦 Dataset

This project uses the [Asian Drama Dataset on Kaggle](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset), including:

- 20,000+ metadata entries for dramas, movies, specials, and TV shows
- Rich features: tags, cast, genres, synopsis, scores, network, release data
- Fully cleaned and preprocessed for NLP tasks

---

## 🧠 Model & Recommendation Strategy

- **Embedding model**: [`sentence-transformers/all-mpnet-base-v2`](https://www.sbert.net/)
- **Text input**: A "soup" combining title, synopsis, genre, tags, cast, creators, etc.
- **Similarity**: Cosine similarity computed using `sentence_transformers.util.cos_sim`
- **Validation**: Ground-truth recommendation pairs used for offline evaluation

---

## 🚀 Installation & Run Locally

```bash
# Clone the repository
git clone https://github.com/lakhindarpal/NextEpisode.git
cd NextEpisode

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

You can also load the precomputed embeddings and data using:

```python
import joblib

media_df = joblib.load("media_df.pkl")
embeddings = joblib.load("embeddings.pkl")
```

---

## 🖼️ UI Preview

> 📌 _Click on a title card to view full information like synopsis, cast, genres, network, episodes, and more._

| 📱 Mobile                              | 💻 Desktop                               |
| -------------------------------------- | ---------------------------------------- |
| ![mobile](screenshots/mobile_view.png) | ![desktop](screenshots/desktop_grid.png) |

---

## 🧪 Evaluation

While the system uses cosine similarity over transformer embeddings, the 80,000 human-labeled positive recommendation pairs can be used for validation using metrics like recall\@K.

---

## ☁️ Deployment

This app is live on [Streamlit Cloud](https://nextepisode.streamlit.app).
You can also deploy with:

- 🔹 Docker
- 🔹 Hugging Face Spaces
- 🔹 Self-hosting with Streamlit Sharing

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) for the frontend
- [SentenceTransformers](https://www.sbert.net/) for embeddings
- [PyTorch](https://pytorch.org/) for tensor ops
- [Joblib](https://joblib.readthedocs.io/) for model persistence
- [Pandas](https://pandas.pydata.org/) for data handling

---

## ⚖ License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset)

---

## 💬 Feedback

Feel free to open an [issue](https://github.com/lakhindarpal/NextEpisode/issues) or drop feedback for feature requests and improvements!
