# Week 4 Project: Retrieval augmented generation

```
pip install -e .
```

This project will explore data-centric approaches to building high quality RAG systems.

### Setup

Please copy the `.env.template` to a `.env` file. 
- Please ask the teaching team for an OpenAI API key to use GPT 3.5. You will set this key to the `OPENAI_API_KEY` variable in `.env`.
- Please ask the teaching team for a Starpoint API key to use the Starpoint vector db. You will set this key to the `STARPOINT_API_KEY` variable in `.env`.

### Project

There is code for you to complete in the following files

- `scripts/build_eval_set.py`
- `scripts/insert_docs.py`
- `scripts/optimizer_params.py`

We recommend you to follow the instructions on Uplimit closely.

### Hyperparameter Optimization Results

| Hit Rate           | Embedding            | Text Search Weight | Hyde Embeddings |
|--------------------|----------------------|--------------------|-----------------|
| 0.9949238578680203 | thenlper/gte-small   | 0.5                | False           |
| 0.9949238578680203 | thenlper/gte-small   | 0.0                | False           |
| 0.9847715736040609 | all-MiniLM-L6-v2     | 0.0                | False           |
| 0.9847715736040609 | all-MiniLM-L6-v2     | 0.5                | False           |
| 0.8984771573604061 | thenlper/gte-small   | 0.0                | True            |
| 0.8984771573604061 | thenlper/gte-small   | 0.5                | True            |
| 0.883248730964467  | all-MiniLM-L6-v2     | 0.0                | True            |
| 0.883248730964467  | all-MiniLM-L6-v2     | 0.5                | True            |


