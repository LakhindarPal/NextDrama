# 🎬 NextDrama

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nextdrama.streamlit.app)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NextDrama** is a content-based recommendation system for Asian dramas, movies, specials, and TV shows. It leverages deep semantic similarity using Sentence Transformers to suggest similar titles based on what you love — no user history required.

> 🌐 **Live Demo:** [nextdrama.streamlit.app](https://nextdrama.streamlit.app)

---

## 🔍 Features

- 🤖 Semantic similarity with `all-mpnet-base-v2` from Sentence Transformers
- 🧠 Content-based recommendations — no collaborative filtering needed
- 🎛️ Filter by genre, country, type, rating, tags, and release year
- 🖼️ Interactive UI with responsive grid layouts
- 🪪 Card-based display with full metadata: cast, synopsis, creators, etc.
- ⚡ Fast cosine similarity search over 20,000+ media embeddings

---

## 📦 Dataset

The system is powered by the [Asian Drama Dataset on Kaggle](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset), which includes:

- Over **20,000** titles with rich metadata
- Fields like title, synopsis, cast, genres, tags, score, country, and network
- Preprocessed and ready for NLP tasks or recommender systems

---

## 🧠 Model & Recommendation Strategy

- **Embedding Model**: [`all-mpnet-base-v2`](https://www.sbert.net/)
- **Input Text**: A combined "soup" of title, synopsis, genres, tags, cast, and directors
- **Similarity Metric**: Cosine similarity via `sentence_transformers.util.cos_sim`
- **Inference**: Fast top-k search with PyTorch tensors
- **Evaluation**: 80,000+ human-labeled positive recommendation pairs for recall@K evaluation

---

## 🚀 Installation & Run Locally

```bash
# Clone the repository
git clone https://github.com/lakhindarpal/NextDrama.git
cd NextDrama

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

You can also manually load the precomputed data:

```python
import joblib

media_df = joblib.load("media_df.pkl")
embeddings = joblib.load("embeddings.pkl")
```

---

## 🖼️ UI Preview

> 📌 Click on a title card to reveal full information including synopsis, cast, tags, genres and more.

| 📱 Mobile                              | 💻 Desktop                               |
| -------------------------------------- | ---------------------------------------- |
| ![mobile](screenshots/mobile_view.png) | ![desktop](screenshots/desktop_grid.png) |

---

## 🧪 Evaluation

The current model uses cosine similarity over transformer-based embeddings. Evaluation can be performed offline using the 80K labeled recommendation pairs and metrics like **Recall\@K**.

---

## ☁️ Deployment

Currently live on **Streamlit Cloud**:
👉 [nextdrama.streamlit.app](https://nextdrama.streamlit.app)

You can also deploy on:

- 🐳 Docker
- 🤗 Hugging Face Spaces
- 🔧 Self-hosting or private servers

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) — frontend framework
- [SentenceTransformers](https://www.sbert.net/) — semantic embeddings
- [PyTorch](https://pytorch.org/) — tensor computation
- [Joblib](https://joblib.readthedocs.io/) — model/data persistence
- [Pandas](https://pandas.pydata.org/) — data handling

---

## ⚖ License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙌 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/lakhindarpal/asian-drama-dataset)

---

## 💬 Feedback

Found a bug? Have a suggestion?
Open an [issue](https://github.com/lakhindarpal/NextDrama/issues) or drop feedback for improvements!
