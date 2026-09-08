---
title: Smart MCQ Solver
emoji: "🧠"
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Smart MCQ Solver

An interactive Hugging Face Space for ranking the five answer choices of a multiple-choice question. It uses the same DeBERTa multiple-choice inference format as the training notebook and returns the top three labels.

## Deploy to Hugging Face Spaces

1. Export and upload a **fine-tuned** `AutoModelForMultipleChoice` checkpoint to a Hugging Face model repository. The model repository must contain `config.json`, model weights (`model.safetensors` or `pytorch_model.bin`), tokenizer files, and (for DeBERTa-v3) the SentencePiece tokenizer model.
2. Create a new Space, select **Gradio**, then push this repository to the Space. The metadata at the top of this README tells Spaces to run `app.py`.
3. In the Space settings, add the variable `HF_MODEL_ID` with the model repository ID, for example `your-account/smart-mcq-deberta`. For a private model, add `HF_TOKEN` as a Space secret with read permission.
4. Choose a CPU Basic Space for a compact checkpoint, or a GPU hardware tier for faster startup and inference. The model loads once when the first request arrives.

`vishallllllllllll/smart-mcq-deberta` is used by default. Set `HF_MODEL_ID` only to override it with another compatible fine-tuned checkpoint.

## Exporting from the training notebook

After selecting the best trained fold (or your final model), save its model and tokenizer:

```python
trainer.save_model("smart-mcq-deberta")
tokenizer.save_pretrained("smart-mcq-deberta")
```

Upload that directory to a Hugging Face model repository, then configure the Space with its repository ID. The Space intentionally keeps model weights out of this application repository, which makes builds faster and avoids committing large binary files.

## Local run

```bash
python -m pip install -r requirements.txt
$env:HF_MODEL_ID = "your-account/smart-mcq-deberta"  # PowerShell
python app.py
```
