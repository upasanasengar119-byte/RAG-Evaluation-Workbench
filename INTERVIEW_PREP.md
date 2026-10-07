# Interview prep — RAG Evaluation Workbench

## 60-second pitch

I built a RAG retrieval evaluation workbench. It runs the **same golden questions** against a top-1 baseline and a top-k candidate, then reports mean recall, MRR, and whether unanswerable questions correctly abstain. That lets me prove whether a retrieval change actually helped — instead of shipping vibes.

## Live demo commands

```bash
python demo.py --k 3
python demo.py --k 1
python -m unittest discover -s tests -v
```

## Architecture talking points

- Evaluate the **actual** retriever used in production (not a proxy).
- Keep unanswerable cases out of recall/MRR averages.
- Paired comparison on identical labels reduces dataset confusion.
- Retrieval success ≠ answer correctness (generation can still hallucinate).

## Likely interview Q&A

**Q: Recall vs precision?**  
A: Recall = did we find the needed evidence? Precision = how much junk did we also return? Raising k can help recall while hurting precision/context quality.

**Q: What does MRR emphasize?**  
A: Early ranking of the first relevant doc. Great for users who only read top hits; not enough alone for multi-doc questions.

**Q: Why separate unanswerable cases?**  
A: Recall is undefined with an empty relevance set. Desired behavior is abstain — measure that explicitly.

**Q: Can retrieval eval certify RAG answers?**  
A: No. Good retrieval is necessary but not sufficient. Score answer support / faithfulness separately.

**Q: What would you add in production?**  
A: Versioned corpora + judgments, latency/cost, confidence intervals, slice analysis, held-out blind set after tuning, and a separate answer-faithfulness stage.

## Failure cases to mention

- Broken golden IDs for a tenant → fail the experiment (don’t silently lower scores)
- Recall up, answers worse → extra irrelevant context distracting the generator
- Tiny handcrafted golden set → smoke test only; add paraphrases and multi-doc cases
