import streamlit as st
from model_utils import (
    ngram_predict_next,
    transformer_predict_next,
    hf_status,
)

st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("Next Word Prediction")
st.caption("N-gram baseline vs Pretrained Transformer")

# show HF model availability
status = hf_status()
if not status.get("packages_installed", False):
    st.warning(
        "Transformer packages not installed — falling back to N-gram. Install `transformers` and `torch` to enable."
    )
elif not status.get("model_loaded", False):
    st.info("Transformer packages installed; model will load on first use.")
else:
    st.info(f"Transformer loaded on {status.get('device')}")

text = st.text_input("Enter text prefix")
top_k = st.slider("Top-K predictions", 1, 10, 5)
model_choice = st.selectbox(
    "Select model",
    ["Both", "N-gram", "Transformer"]
)

# Transformer method controls
use_generation = st.checkbox("Use generation (whole-word outputs)", value=True)
gen_cols = None
if use_generation:
    gen_cols = st.columns(2)
    with gen_cols[0]:
        num_beams = st.slider("Beam width", 2, 8, 4)
    with gen_cols[1]:
        max_new_tokens = st.slider("Max new tokens", 1, 50, 10)
else:
    num_beams = 4
    max_new_tokens = 10

if text:
    st.divider()

    if model_choice in ["Both", "N-gram"]:
        st.subheader("N-gram Model")
        preds = ngram_predict_next(text, top_k)
        if preds:
            for w, p in preds:
                st.write(f"**{w}** → {p:.4f}")
        else:
            st.warning("No prediction (data sparsity).")

    if model_choice in ["Both", "Transformer"]:
        st.subheader("Transformer Model")
        preds = transformer_predict_next(
            text,
            top_k=top_k,
            prefer_generation=use_generation,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        for w, p in preds:
            st.write(f"**{w}** → {p:.4f}")
