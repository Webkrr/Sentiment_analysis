# =======================================================
# 🎬 IMDB Sentiment Analysis - Streamlit App with SHAP
# =======================================================

import streamlit as st
import pickle
import nltk
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# ----------------------------------------
# Load Trained Model and TF-IDF Vectorizer
# ----------------------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------------------
# Text Preprocessing
# ----------------------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, and stopwords."""
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict_sentiment(review: str):
    """Predict sentiment and return probabilities and TF-IDF vector."""
    review_clean = clean_text(review)
    review_tfidf = tfidf.transform([review_clean])
    proba = model.predict_proba(review_tfidf)[0]
    pred = model.predict(review_tfidf)[0]
    sentiment = "😊 Positive" if pred == 1 else "☹️ Negative"
    return sentiment, proba, review_tfidf

# ----------------------------------------
# Streamlit Setup
# ----------------------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

st.title("🎬 IMDB Sentiment Analysis Dashboard")
st.write("Analyze movie reviews, explore SHAP explanations, and test the model interactively!")
st.markdown("---")

tabs = st.tabs(["📝 Predict Sentiment", "💬 Example Reviews", "🧠 SHAP Explanation"])

# ------------------------------------------------
# TAB 1 — Predict Sentiment
# ------------------------------------------------
with tabs[0]:
    st.subheader("Enter your movie review:")
    user_input = st.text_area("📝 Review Text:", height=150, placeholder="Type your review here...")

    if st.button("🔍 Predict Sentiment"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            sentiment, proba, review_tfidf = predict_sentiment(user_input)

            st.markdown(f"### Prediction: **{sentiment}**")
            st.bar_chart(pd.DataFrame({
                "Confidence": [proba[0], proba[1]]
            }, index=["☹️ Negative", "😊 Positive"]))

            # Save for SHAP tab
            st.session_state["last_review"] = user_input
            st.session_state["last_vector"] = review_tfidf

# ------------------------------------------------
# TAB 2 — Example Reviews
# ------------------------------------------------
with tabs[1]:
    st.subheader("Try Example Reviews")

    examples = {
        "Amazing movie with great acting!": "Expected: 😊 Positive",
        "Horrible film, waste of time.": "Expected: ☹️ Negative",
        "Not bad, but not great either.": "Expected: Neutral/Moderate",
        "I absolutely loved the cinematography!": "Expected: 😊 Positive",
        "I hate it.": "Expected: ☹️ Negative",
    }

    for text, label in examples.items():
        if st.button(text):
            sentiment, proba, review_tfidf = predict_sentiment(text)
            st.write(f"**{label}** → Model Prediction: **{sentiment}**")
            st.bar_chart(pd.DataFrame({
                "Confidence": [proba[0], proba[1]]
            }, index=["☹️ Negative", "😊 Positive"]))

            st.session_state["last_review"] = text
            st.session_state["last_vector"] = review_tfidf

# ------------------------------------------------
# TAB 3 — SHAP Explanation
# ------------------------------------------------
with tabs[2]:
    st.subheader("Model Explanation with SHAP")
    st.write("Understand which words most influenced the model’s prediction.")

    if "last_review" in st.session_state:
        review_text = st.session_state["last_review"]
        review_tfidf = st.session_state["last_vector"]

        try:
            # Use zero baseline (neutral reference)
            background = np.zeros((1, review_tfidf.shape[1]))
            explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
            shap_values = explainer.shap_values(review_tfidf)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if hasattr(shap_values, "toarray"):
                shap_values = shap_values.toarray()

            feature_names = tfidf.get_feature_names_out()

            # Show only words that appear in the review
            nonzero_idx = review_tfidf.nonzero()[1]
            words_in_review = [feature_names[i] for i in nonzero_idx]
            shap_for_present_words = shap_values[0, nonzero_idx]

            shap_df = pd.DataFrame({
                "Word": words_in_review,
                "Impact": shap_for_present_words
            }).sort_values(by="Impact", ascending=False)

            # Pick top contributors by absolute value
            shap_df = shap_df.reindex(shap_df["Impact"].abs().sort_values(ascending=False).index).head(15)

            st.write("### 🔍 Top Words Influencing Prediction")
            st.bar_chart(shap_df.set_index("Word"))

            st.caption(f"Explanation for review: *'{review_text}'*")

        except Exception as e:
            st.error(f"⚠️ SHAP could not generate explanation: {e}")
            st.info("Try using a slightly longer or more detailed review.")
    else:
        st.info("Run a prediction first in the 'Predict Sentiment' tab to view SHAP explanations.")

st.markdown("---")
st.caption("Built using Streamlit, Scikit-learn, NLTK, and SHAP.")
