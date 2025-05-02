import streamlit as st
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt_tab")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Load AG News and extract sentences
@st.cache_data
def load_ag_news_sentences(max_sentences=5000):
    dataset = load_dataset("ag_news", split="train[:5%]")
    texts = [x["text"] for x in dataset]

    sentences = []
    for text in texts:
        sents = sent_tokenize(text)
        sentences.extend(sents)

    sentences = [s for s in sentences if len(s.split()) > 3]
    return sentences[:max_sentences]

# Compute sentence embeddings
@st.cache_data
def embed_sentences(sentences):
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Load your trained LCM model
class PreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.linear(self.norm(x)))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, hidden_dim))

    def forward(self, x):
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        x = x + self.pos_embed[:, :seq_len]
        for layer in self.layers:
            x = layer(x, x, tgt_mask=mask)
        return x


class PostNet(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Parameter(torch.ones(output_dim))
        self.shift = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return self.linear(x) * self.scale + self.shift


class BaseLCM(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, num_heads=8, num_layers=6, ff_dim=2048):
        super().__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
        self.postnet = PostNet(hidden_dim, input_dim)

    def forward(self, x):
        x = self.prenet(x)
        x = self.decoder(x)
        return self.postnet(x)

# Load model weights
@st.cache_resource
def load_lcm_model(path="base_lcm_model.pth"):
    model = BaseLCM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Find the most similar sentence to the predicted embedding
def find_similar_sentence(pred_embedding, all_embeddings, all_sentences, top_k=1):
    # Normalize embeddings
    pred_embedding = pred_embedding / np.linalg.norm(pred_embedding)
    
    # Calculate similarities
    similarities = cosine_similarity([pred_embedding], all_embeddings)[0]
    
    # Get top k most similar sentences
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_sentences = [all_sentences[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    
    return list(zip(top_sentences, top_scores))

# App UI
st.title("Large Concept Model Demo (AG News)")

# Load sentences and precompute embeddings
sentences = load_ag_news_sentences()
embeddings = embed_sentences(sentences)
model = load_lcm_model()

input_text = st.text_area("Enter a sentence (or few sentences):", height=150)

if st.button("Predict Next Sentence"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing input..."):
            # Tokenize input into sentences
            input_sents = sent_tokenize(input_text)
            
            # Keep last 9 sentences (or fewer if input has fewer)
            input_sents = input_sents[-9:] if len(input_sents) > 9 else input_sents
            
            if len(input_sents) < 1:
                st.warning("Please enter at least one sentence.")
            else:
            
                               
                # Pad with empty strings if needed to get seq_len=9
                while len(input_sents) < 9:
                    input_sents.insert(0, "")
                
                # Encode input sentences
                input_embeds = embedder.encode(input_sents, convert_to_tensor=True).unsqueeze(0)
                
                # Generate prediction
                with torch.no_grad():
                    pred_embed = model(input_embeds)[:, -1, :].squeeze().cpu().numpy()
                
                # Find most similar sentences in our dataset
                similar_sentences = find_similar_sentence(pred_embed, embeddings, sentences)
                
                # Display results
                st.subheader("Predicted Next Sentences:")
                for i, (sentence, score) in enumerate(similar_sentences):
                    
                    st.write(sentence)
                    st.write("---")