"""Retrieval evaluation: precision@k for k in (1, 3, 5) over eval_queries.json."""
import json
from ask import retrieve

KS = (1, 3, 5)


def is_relevant(chunk, q):
    if chunk["source"] != q["source"]:
        return False
    text = chunk["text"].lower()
    return chunk["page"] in q["relevant_pages"] or any(k.lower() in text for k in q["keywords"])


def main():
    queries = json.load(open("eval_queries.json"))
    rows = []
    for q in queries:
        chunks = retrieve(q["question"], k=max(KS))
        hits = [is_relevant(c, q) for c in chunks]
        rows.append({"question": q["question"], "source": q["source"],
                     **{f"p@{k}": sum(hits[:k]) / k for k in KS}})
    means = {f"p@{k}": sum(r[f"p@{k}"] for r in rows) / len(rows) for k in KS}

    print(f"{'#':<3}{'p@1':>6}{'p@3':>6}{'p@5':>6}  question")
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}{r['p@1']:>6.2f}{r['p@3']:>6.2f}{r['p@5']:>6.2f}  {r['question'][:70]}")
    print(f"{'avg':<3}{means['p@1']:>6.2f}{means['p@3']:>6.2f}{means['p@5']:>6.2f}")

    json.dump({"mean": means, "per_question": rows}, open("eval_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
