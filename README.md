# 🧠 Large Concept Model (LCM)

**LCM** is an autoregressive sentence-level language model inspired by Meta's research on Large Concept Models. Instead of predicting the next word, LCM operates in the semantic embedding space of entire sentences, enabling concept-level generation and understanding.

---

## ✨ Key Features

- 📚 Trained on [WikiText-103](https://huggingface.co/datasets/wikitext), a high-quality dataset of Wikipedia articles.
- 🔗 Uses [Sentence-BERT](https://www.sbert.net/) embeddings (`all-MiniLM-L6-v2`) for semantic encoding.
- 🧩 Transformer-based autoregressive decoder predicts the next sentence embedding.
- 🔍 Retrieves the closest real sentence using cosine similarity from AG News dataset.
- 🖥️ Interactive Streamlit demo for real-time sentence generation.

---

## 🏗️ Architecture

- **PreNet**: Normalizes and projects sentence embeddings.
- **Transformer Decoder**: Autoregressively models sentence embedding sequences.
- **PostNet**: Projects hidden states back to sentence embedding space.
- **Retrieval**: Finds the closest matching sentence from a reference dataset.

---

## 🚀 How It Works

1. **Input**: User enters a paragraph (1–9 sentences) into the Streamlit interface.
2. **Encoding**: Sentences are embedded using `all-MiniLM-L6-v2` (384-dim).
3. **Prediction**: Transformer decoder autoregressively predicts the next sentence embedding.
4. **Retrieval**: Finds the most similar sentence from AG News dataset using cosine similarity.
5. **Output**: Displays the top-matching sentence as the likely continuation.
