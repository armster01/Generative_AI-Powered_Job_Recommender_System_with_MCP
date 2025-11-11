# Generative AI-Powered Job Recommender System with MCP

A Python-based job recommender system that combines generative AI with a Model Context Protocol (MCP) to produce personalized, explainable job recommendations. This repository contains code, examples, and tools for data preprocessing, model training/inference, evaluation, and a lightweight API to serve recommendations.

> Note: "MCP" in the repository title refers to the project's modular component pipeline used to compose data processing, candidate ranking, and generative explanation modules. If you prefer a different expansion for MCP, replace it in the docs.

Status: Prototype / Research — adapt and extend for production.

# Features
- Candidate generation and ranking using ML and retrieval techniques
- Generative explanation module to create clear, personalized rationales for each recommended job
- Model Context Protocol (MCP) to mix & match components: featurizers, retrievers, rankers, and explainers
- Evaluation metrics and utilities for offline testing
- Example scripts for training, inference, and running a REST API

# Table of contents
- Project structure
- Quick start
- Installation
- Configuration & environment variables
- Data preparation
- Training
- Running inference / API
- Evaluation
- Development notes
- Contributing
- License
- Contact

# Project structure (suggested)
- data/                 — raw & processed datasets (not checked in)
- src/                  — Python package with model, pipeline, and API code
  - src/pipeline/       — MCP components (retrievers, rankers, explainers)
  - src/models/         — model definitions & wrappers (ML/ranking/generative)
  - src/utils/          — helpers: preprocessing, metrics, persistence
  - src/api/            — FastAPI/Flask server to serve recommendations
- notebooks/            — EDA and model prototyping
- tests/                — unit/integration tests
- requirements.txt
- README.md
- configs/              — sample config files

# Quick start (local development)
1. Clone the repo
   ```bash
   git clone https://github.com/armster01/Generative_AI-Powered_Job_Recommender_System_with_MCP.git
   cd Generative_AI-Powered_Job_Recommender_System_with_MCP
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\activate    # Windows (PowerShell)
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a requirements file yet, typical packages include:
   - fastapi / uvicorn
   - transformers / sentence-transformers
   - scikit-learn
   - pandas / numpy
   - faiss-cpu (or faiss-gpu)
   - torch / tensorflow (depending on models)
   - pytest

# Configuration & environment variables
- OPENAI_API_KEY: (optional) for using OpenAI generative models
- APIFY_API_TOKEN: for extracting the info from NAUKRI & LINKEDIN
- HF_API_TOKEN: (optional) for Hugging Face private models
- MLFLOW_TRACKING_URI: (optional) if using MLFlow for experiments
- DATABASE_URL: (optional) for persistence

Set variables in your shell or a .env file. Example:
```bash
export OPENAI_API_KEY="sk-..."
export HF_API_TOKEN="hf_..."
```

# Data preparation
1. Place raw job postings and user interactions in data/raw/.
2. Run preprocessing to create features and candidate indexes:
   ```bash
   python -m src.utils.preprocess --input data/raw/jobs.csv --output data/processed/jobs.parquet
   python -m src.pipeline.build_index --jobs data/processed/jobs.parquet --index-dir data/indexes/
   ```
3. Example scripts/notebooks show expected formats (job_id, title, description, skills, location, posted_date) and user events (user_id, job_id, event_type, timestamp).

# Training
- Train ranking models or fine-tune encoders:
  ```bash
  python -m src.models.train_ranker --config configs/ranker.yaml
  ```
- Fine-tune a sentence encoder for retrieval:
  ```bash
  python -m src.models.train_encoder --data data/processed/train.parquet --output models/encoder
  ```

# Inference (CLI)
- Run a simple offline inference to get top-K recommendations:
  ```bash
  python -m src.pipeline.recommend --user-id 123 --top-k 10 --index data/indexes/ --model models/ranker
  ```
- Output is a JSON/CSV with recommended job ids plus scores and generative explanations (if enabled).

# Run API (example using FastAPI)
1. Start the API server (replace entrypoint if different):
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Example HTTP request:
   POST /recommend
   ```json
   {
     "user_id": 123,
     "context": {
       "current_title": "Data Analyst",
       "skills": ["sql", "python", "tableau"]
     },
     "top_k": 5,
     "explain": true
   }
   ```
   Response includes recommended jobs, ranking scores, and natural-language explanations per job.

# Generative explanation module
- The explainer composes personalized rationales using:
  - user profile signals (skills, title, history)
  - job posting content and tags
  - ranking features and score breakdown
- It can use a local model (Transformers) or an external API (OpenAI/Hugging Face). Configure API keys in environment variables.

# Evaluation
- Offline metrics supported:
  - Ranking: NDCG@k, MAP, Precision@k, Recall@k
  - Diversity & novelty metrics
  - Explainability: optional human evaluation or automated heuristics
- Run evaluations:
  ```bash
  python -m src.utils.evaluate --predictions outputs/preds.json --ground-truth data/processed/gt.parquet --metrics ndcg@10,precision@5
  ```

# Development notes
- Keep components modular:
  - Retriever: candidate generation using vector similarity or lexical filters
  - Ranker: supervised model combining features
  - Explainer: template-based or generative model
- Add unit tests to tests/ for each component
- Use small sample data in data/samples/ for rapid iteration

# Best practices & tips
- Use embeddings & ANN (FAISS) for scalable retrieval; keep index sharded for large corpora
- Log experiments (MLflow/Weights & Biases) and persist model artifacts
- Sanitize inputs to generative models to avoid hallucinations; include grounding context
- Constrain explanation length and ensure sensitive information is not leaked

# Contributing
- Found a bug or want a feature? Open an issue with:
  - Reproduction steps
  - Minimal dataset / sample input
  - Error logs
- Pull requests: fork → feature branch → open PR. Keep changes small and include tests.
- Add or update docs in docs/ and example notebooks.

# Acknowledgements
- Inspired by recommender-system and explainable-AI research.
- Uses open-source libraries: Hugging Face, Faiss, scikit-learn, FastAPI, etc.

# Contact
- Maintainer: armster01
- For questions or collaboration ideas, open an issue or contact me via GitHub profile.

# What to add next
- Dockerfile + docker-compose for one-command runs
- CI pipeline for tests and linting
- Example dataset and reproducible experiment notebook
- More robust evaluation harness (A/B simulation, online metrics)

# If you'd like, I can:
- Generate a starter requirements.txt and sample config files
- Create example notebook(s) for data ingestion and training
- Draft Dockerfile and a small FastAPI app scaffold
Tell me which you want next and I'll prepare the files.

<img width="1893" height="810" alt="Screenshot 2025-11-11 202224" src="https://github.com/user-attachments/assets/841cd616-0994-452a-b754-5a8a12ed4a08" />
