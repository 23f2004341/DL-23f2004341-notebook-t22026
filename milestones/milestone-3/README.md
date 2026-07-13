# Milestone 3 – Retrieval-Augmented Generation (RAG)

## Objective

The objective of Milestone 3 is to understand and implement Retrieval-Augmented Generation (RAG) pipelines for improving reasoning in multiple-choice question answering. This milestone explores semantic retrieval, vector databases, reranking, and context augmentation using transformer-based models.

---

## Topics Covered

- Retrieval-Augmented Generation (RAG)
- Dense Vector Embeddings
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS Vector Database
- Top-k Document Retrieval
- Cross-Encoder Re-ranking
- Context Augmentation
- Zero-shot Classification with Retrieved Context
- Adversarial RAG
- Retrieval Evaluation (Hit Rate)

---

## Tasks Completed

- Generated dense embeddings for knowledge base documents.
- Built a FAISS vector index for semantic retrieval.
- Retrieved top-k relevant documents using cosine similarity.
- Evaluated retrieval quality using document ranking.
- Applied Cross-Encoder reranking with `ms-marco-MiniLM-L-6-v2`.
- Constructed RAG prompts by combining retrieved context with questions.
- Performed zero-shot classification on augmented prompts.
- Evaluated the impact of incorrect retrieval using adversarial context.
- Calculated retrieval Hit Rate over multiple samples.

---

## Technologies Used

- Python
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- Cross-Encoder
- PyTorch
- Scikit-learn

---

## Learning Outcomes

After completing this milestone, the following concepts were understood:

- Dense semantic retrieval
- Vector database construction
- Approximate nearest neighbor search
- Cross-Encoder reranking
- Retrieval-Augmented Generation
- Context-aware reasoning
- Retrieval evaluation metrics

---


## Next Milestone

Milestone 4 focuses on fine-tuning transformer models using LoRA and full parameter optimization for the Smart MCQ Solver Challenge.