"""Streamlit UI for the Smart MCQ DeBERTa model."""

import os

import streamlit as st
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer


MODEL_ID = os.getenv("HF_MODEL_ID", "vishallllllllllll/smart-mcq-deberta")
OPTIONS = ("A", "B", "C", "D", "E")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "320"))


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def rank_answers(question: str, answers: list[str]):
    tokenizer, model = load_model()
    encoded = tokenizer(
        [question] * len(OPTIONS),
        answers,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.unsqueeze(0) for key, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits[0], dim=-1).tolist()
    return sorted(zip(OPTIONS, probabilities), key=lambda item: item[1], reverse=True)


st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")
st.title("🧠 Smart MCQ Solver")
st.caption("Rank five answer choices with the fine-tuned DeBERTa multiple-choice model.")

with st.form("mcq_form"):
    question = st.text_area("Question", height=130, placeholder="Enter the multiple-choice question")
    answers = [st.text_input(f"Option {letter}", key=f"option_{letter}") for letter in OPTIONS]
    submitted = st.form_submit_button("Rank answers", type="primary")

if submitted:
    question = question.strip()
    answers = [answer.strip() for answer in answers]
    if not question or any(not answer for answer in answers):
        st.error("Enter a question and text for all five answer options.")
    else:
        try:
            ranking = rank_answers(question, answers)
        except Exception as exc:
            st.error("The model could not be loaded. Please try again shortly.")
            st.exception(exc)
        else:
            st.subheader("Ranking")
            for rank, (letter, probability) in enumerate(ranking, start=1):
                st.progress(probability, text=f"{rank}. Option {letter} — {probability:.1%}")
            st.success("Top 3: " + " → ".join(letter for letter, _ in ranking[:3]))

st.caption(f"Model: {MODEL_ID}")
