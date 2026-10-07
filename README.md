# RAG Evaluation Workbench

Portfolio project: compare retrieval configurations on a **golden question set** using recall, MRR, and correct abstention for unanswerable queries.

## Why this matters

Teams often change chunking / top-k without evidence. This workbench runs a paired experiment:

- **Baseline:** top-1 retrieval
- **Candidate:** top-k retrieval (configurable)
- Same corpus + same labels → inspect which questions improved or got noisier

Scope is **retrieval only** (does not certify LLM answer correctness).

## Quick start

```bash
cd RAG-Evaluation-Workbench
python demo.py
python demo.py --k 1
python -m unittest discover -s tests -v
```

No third-party packages required (Python 3.10+).

## Metrics

| Metric | Meaning |
|--------|---------|
| Recall | Fraction of labeled relevant docs retrieved |
| Precision | Fraction of retrieved docs that are relevant |
| MRR | Rewards early rank of first relevant hit |
| abstained_correctly | For empty relevance labels, empty retrieval is success |

## Project layout

```text
data/documents.json   # multi-tenant corpus
data/golden.json      # labeled queries (+ unanswerable)
src/search.py         # TF-IDF cosine retriever
src/evaluate.py       # paired experiment + metrics
demo.py
INTERVIEW_PREP.md
```

## Architecture

```text
Golden questions + authorized corpus
    → Paired retrieval runs (k=1 vs k=N)
    → Case-level metrics
    → Comparison report (mean recall, MRR, recall_delta)
```

## License

MIT — feel free to fork and extend.
