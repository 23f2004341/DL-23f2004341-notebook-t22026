"""Gradio application for ranking five multiple-choice answers."""

import os
from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer


OPTIONS = ("A", "B", "C", "D", "E")
MODEL_ID = os.getenv("HF_MODEL_ID", "vishallllllllllll/smart-mcq-deberta")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "320"))


@lru_cache(maxsize=1)
def load_model():
    """Download a Hugging Face multiple-choice checkpoint once per Space."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def predict(prompt, option_a, option_b, option_c, option_d, option_e):
    prompt = (prompt or "").strip()
    answers = [option_a, option_b, option_c, option_d, option_e]
    if not prompt or any(not str(answer).strip() for answer in answers):
        raise gr.Error("Enter a question and text for all five answer options.")

    try:
        tokenizer, model = load_model()
    except Exception as exc:
        raise gr.Error(
            f"Could not load '{MODEL_ID}'. Set HF_MODEL_ID to a public fine-tuned "
            "multiple-choice model repository, or configure its access token as a Space secret."
        ) from exc

    inputs = tokenizer(
        [prompt] * len(OPTIONS),
        [str(answer) for answer in answers],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: value.unsqueeze(0) for name, value in inputs.items()}

    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1).tolist()

    ranked = sorted(zip(OPTIONS, probabilities), key=lambda item: item[1], reverse=True)
    scores = {letter: float(score) for letter, score in ranked}
    top_three = " → ".join(letter for letter, _ in ranked[:3])
    return scores, top_three


with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown(
        "# Smart MCQ Solver\n"
        "Rank five answer choices with a fine-tuned DeBERTa multiple-choice model. "
        "The output gives the three most likely answer labels."
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Question", lines=5, placeholder="Enter the multiple-choice question")
            option_a = gr.Textbox(label="A")
            option_b = gr.Textbox(label="B")
            option_c = gr.Textbox(label="C")
            option_d = gr.Textbox(label="D")
            option_e = gr.Textbox(label="E")
            submit = gr.Button("Rank answers", variant="primary")
        with gr.Column():
            scores = gr.Label(label="Probability by option", num_top_classes=5)
            top_three = gr.Textbox(label="Top 3 ranking")
            gr.Markdown(f"Running model: `{MODEL_ID}`")

    submit.click(
        predict,
        inputs=[prompt, option_a, option_b, option_c, option_d, option_e],
        outputs=[scores, top_three],
    )


if __name__ == "__main__":
    demo.launch()
