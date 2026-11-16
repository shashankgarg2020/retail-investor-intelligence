"""Measure the system. Prints real numbers from this database and these label files.

    python -m scripts.evaluate export --per-company 40     # write a labelling sheet
    python -m scripts.evaluate ingestion
    python -m scripts.evaluate relevance
    python -m scripts.evaluate sentiment
    python -m scripts.evaluate retrieval
    python -m scripts.evaluate latency
    python -m scripts.evaluate all

Label files live in data/eval/:
  labels.csv               post_id, ticker, platform, text, human_relevance, human_sentiment
  retrieval_questions.json [{"question": ..., "ticker": ..., "relevant_ids": [...]}]

Rows with an empty human_* cell are skipped, so partial labelling still works.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from config.settings import companies
from src.analytics import metrics
from src.analytics.metrics import Filters
from src.database.connection import get_engine

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
LABELS = EVAL_DIR / "labels.csv"
QUESTIONS = EVAL_DIR / "retrieval_questions.json"
RESULTS = EVAL_DIR / "results.json"     # read by the API's /quality endpoint and the dashboard
SENTIMENTS = ["positive", "neutral", "negative"]
RESULTS_OUT: dict = {}


def _wide_filters(ticker: str) -> Filters:
    now = datetime.now(timezone.utc)
    return Filters(ticker, now - timedelta(days=3650), now + timedelta(days=1))


def export(per_company: int) -> None:
    """Sample posts for hand labelling. Model labels are deliberately left out of the sheet."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if LABELS.exists():
        raise SystemExit(f"{LABELS} already exists; move it first so labels are not overwritten.")
    rows = []
    with get_engine().connect() as conn:
        for ticker in companies():
            # Stratified by the system's relevance label so all three outcomes are represented,
            # then shuffled with a fixed seed so the export is repeatable.
            found = conn.execute(text("""
                SELECT id, ticker, platform, body FROM (
                    SELECT p.id, c.ticker, p.platform,
                           coalesce(p.title || '. ', '') || p.text AS body,
                           row_number() OVER (PARTITION BY a.relevance_label ORDER BY md5(p.id::text)) AS rn
                    FROM posts p JOIN companies c ON c.id = p.company_id
                    JOIN post_analysis a ON a.post_id = p.id
                    WHERE c.ticker = :t AND p.is_synthetic = FALSE AND p.language = 'en'
                      AND a.duplicate_of IS NULL
                ) s ORDER BY rn, md5(id::text) LIMIT :n
            """), {"t": ticker, "n": per_company})
            rows += [[r.id, r.ticker, r.platform, r.body[:700], "", ""] for r in found]
    with open(LABELS, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["post_id", "ticker", "platform", "text", "human_relevance", "human_sentiment"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {LABELS}")
    print("Fill human_relevance with relevant / irrelevant, and human_sentiment with "
          "positive / neutral / negative (sentiment only for relevant posts).")


def _load_labels() -> list[dict[str, str]]:
    if not LABELS.exists():
        raise SystemExit(f"No label file at {LABELS}. Run: python -m scripts.evaluate export")
    with open(LABELS, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _system_labels(ids: list[int]) -> dict[int, dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT post_id, relevance_label, relevance_method, sentiment, sentiment_model,
                   finbert_sentiment
            FROM post_analysis WHERE post_id = ANY(:ids)
        """), {"ids": ids})
        return {r.post_id: dict(r._mapping) for r in rows}


def ingestion() -> None:
    print("\n== Ingestion (from ingestion_runs)")
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT platform, count(*) AS runs,
                   count(*) FILTER (WHERE status = 'success') AS ok_runs,
                   sum(requests_ok) AS req_ok, sum(requests_failed) AS req_failed,
                   sum(fetched) AS fetched, sum(inserted) AS inserted, sum(duplicates) AS duplicates,
                   round(avg(extract(epoch FROM completed_at - started_at))::numeric, 1) AS avg_seconds
            FROM ingestion_runs WHERE is_synthetic = FALSE GROUP BY platform ORDER BY platform
        """)).all()
        totals = conn.execute(text("""
            SELECT count(*) AS posts, count(*) FILTER (WHERE is_synthetic) AS synthetic,
                   (SELECT count(*) FROM post_analysis WHERE duplicate_of IS NOT NULL) AS text_duplicates
            FROM posts
        """)).one()
    if not rows:
        print("   no real ingestion runs recorded")
    for r in rows:
        requests = (r.req_ok or 0) + (r.req_failed or 0)
        success = f"{100 * r.req_ok / requests:.1f}%" if requests else "n/a"
        dup = f"{100 * r.duplicates / r.fetched:.1f}%" if r.fetched else "n/a"
        print(f"   {r.platform:8} runs {r.ok_runs}/{r.runs} fully successful | request success {success} "
              f"({r.req_ok}/{requests}) | fetched {r.fetched}, new {r.inserted}, "
              f"platform-id duplicates {dup} | avg run {r.avg_seconds}s")
    print(f"   source coverage: {len(rows)}/3 platforms with real runs")
    print(f"   stored posts: {totals.posts} ({totals.synthetic} synthetic); "
          f"text-level duplicates flagged: {totals.text_duplicates}")
    RESULTS_OUT["ingestion"] = {
        "platforms": {r.platform: {"runs": r.runs, "requests_ok": r.req_ok, "requests_failed": r.req_failed,
                                   "fetched": r.fetched, "inserted": r.inserted} for r in rows},
        "stored_posts": totals.posts, "synthetic_posts": totals.synthetic,
        "text_duplicates": totals.text_duplicates,
    }


def _labeller() -> str:
    who = {r.get("labelled_by", "").strip() for r in _load_labels() if r.get("labelled_by")}
    return ", ".join(sorted(who)) or "human annotator"


def relevance() -> None:
    from sklearn.metrics import precision_recall_fscore_support
    rows = [r for r in _load_labels() if r["human_relevance"].strip()]
    system = _system_labels([int(r["post_id"]) for r in rows])
    rows = [r for r in rows if int(r["post_id"]) in system]
    print(f"\n== Entity resolution ({len(rows)} labelled posts; labels by: {_labeller()})")
    if not rows:
        return
    truth = [r["human_relevance"].strip().lower() == "relevant" for r in rows]
    out: dict = {"n": len(rows), "labelled_by": _labeller(), "per_company": {}}
    for key, name, positive in (("strict", "strict: only 'relevant' counts", {"relevant"}),
                                ("lenient", "lenient: 'possibly_relevant' also counts",
                                 {"relevant", "possibly_relevant"})):
        predicted = [system[int(r["post_id"])]["relevance_label"] in positive for r in rows]
        p, rec, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        print(f"   {name}: precision {p:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
        out[key] = {"precision": round(p, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    for ticker in companies():
        subset = [(t, system[int(r["post_id"])]["relevance_label"] == "relevant")
                  for t, r in zip(truth, rows) if r["ticker"] == ticker]
        if subset:
            p, rec, f1, _ = precision_recall_fscore_support(*zip(*subset), average="binary", zero_division=0)
            print(f"   {ticker} (n={len(subset)}): precision {p:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
            out["per_company"][ticker] = {"n": len(subset), "precision": round(p, 3),
                                          "recall": round(rec, 3), "f1": round(f1, 3)}
    undecided = sum(system[int(r["post_id"])]["relevance_label"] == "possibly_relevant" for r in rows)
    print(f"   left undecided (possibly_relevant): {undecided}")
    out["undecided"] = undecided
    RESULTS_OUT["relevance"] = out


def _score(truth: list[str], predicted: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(truth, predicted, labels=SENTIMENTS, zero_division=0)
    return {
        "n": len(truth),
        "accuracy": round(accuracy_score(truth, predicted), 3),
        "macro_f1": round(f1_score(truth, predicted, average="macro", labels=SENTIMENTS, zero_division=0), 3),
        "per_class": {lab: {"precision": round(p[i], 3), "recall": round(r[i], 3), "f1": round(f[i], 3),
                            "support": int(s[i])} for i, lab in enumerate(SENTIMENTS)},
        "confusion": confusion_matrix(truth, predicted, labels=SENTIMENTS).tolist(),
    }


def _print_score(name: str, score: dict) -> None:
    classes = "  ".join(f"{lab} P {v['precision']:.2f}/R {v['recall']:.2f}" for lab, v in score["per_class"].items())
    print(f"   {name:26} accuracy {score['accuracy']:.3f}   macro F1 {score['macro_f1']:.3f}   {classes}")
    print("      confusion (rows = label, columns = model; positive, neutral, negative): "
          + "  ".join(str(row) for row in score["confusion"]))


def sentiment() -> None:
    rows = [r for r in _load_labels() if r["human_sentiment"].strip().lower() in SENTIMENTS]
    system = _system_labels([int(r["post_id"]) for r in rows])
    scored = [r for r in rows if system.get(int(r["post_id"]), {}).get("sentiment")]
    # The head-to-head uses exactly the same posts: those FinBERT and the LLM both scored.
    both = [r for r in scored if system[int(r["post_id"])]["finbert_sentiment"]
            and (system[int(r["post_id"])]["sentiment_model"] or "").startswith("llm:")]
    print(f"\n== Sentiment (labels by: {_labeller()})")
    out: dict = {"labelled_by": _labeller()}
    if scored:
        truth = [r["human_sentiment"].strip().lower() for r in scored]
        majority = max(SENTIMENTS, key=truth.count)
        out["majority_baseline"] = {"label": majority, "accuracy": round(truth.count(majority) / len(truth), 3)}
        out["production"] = _score(truth, [system[int(r["post_id"])]["sentiment"] for r in scored])
        print(f"   {len(scored)} labelled posts scored by the system; "
              f"always-'{majority}' baseline accuracy {out['majority_baseline']['accuracy']:.3f}")
        _print_score("label in use (production)", out["production"])
    if both:
        truth = [r["human_sentiment"].strip().lower() for r in both]
        out["finbert"] = _score(truth, [system[int(r["post_id"])]["finbert_sentiment"] for r in both])
        out["llm"] = _score(truth, [system[int(r["post_id"])]["sentiment"] for r in both])
        print(f"   head to head on the same {len(both)} posts:")
        _print_score("FinBERT", out["finbert"])
        _print_score("LLM investor stance", out["llm"])
        if "claude" in _labeller().lower():
            out["bias_note"] = "Labels were drafted by an LLM, which biases this comparison toward the LLM."
            print(f"   NOTE: {out['bias_note']}")
    RESULTS_OUT["sentiment"] = out


def retrieval(k: int = 5) -> None:
    from src.intelligence.embeddings import embed_query
    from src.rag import retriever
    from src.rag.planner import plan
    if not QUESTIONS.exists():
        raise SystemExit(f"No question file at {QUESTIONS}.")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = [q for q in questions if q.get("relevant_ids")]
    print(f"\n== Retrieval ({len(questions)} questions with marked relevant posts, K={k})")
    if not questions:
        return
    scores: dict[str, dict[str, list[float]]] = {
        m: {"precision": [], "recall": [], "mrr": []} for m in ("keyword only", "vector only", "hybrid (RRF)")}
    with get_engine().connect() as conn:
        for q in questions:
            f = _wide_filters(q["ticker"])
            p = plan(q["question"], q["ticker"], f.start, f.end)
            relevant = set(q["relevant_ids"])
            keyword = retriever.keyword_search(conn, f, p.keywords)
            vector = [i for i, _ in retriever.vector_search(conn, f, embed_query(q["question"]))]
            fused = retriever.rrf([keyword, vector])
            ranked = {"keyword only": keyword, "vector only": vector,
                      "hybrid (RRF)": sorted(fused, key=fused.get, reverse=True)}
            for method, ids in ranked.items():
                top = ids[:k]
                hits = [i for i in top if i in relevant]
                scores[method]["precision"].append(len(hits) / k)
                scores[method]["recall"].append(len(hits) / len(relevant))
                first = next((rank for rank, i in enumerate(ids, 1) if i in relevant), None)
                scores[method]["mrr"].append(1 / first if first else 0.0)
    out: dict = {"questions": len(questions), "k": k, "methods": {}}
    for method, s in scores.items():
        row = {m: round(statistics.mean(v), 3) for m, v in s.items()}
        out["methods"][method] = row
        print(f"   {method:14} precision@{k} {row['precision']:.3f}   "
              f"recall@{k} {row['recall']:.3f}   MRR {row['mrr']:.3f}")
    RESULTS_OUT["retrieval"] = out


def latency(repeats: int = 5) -> None:
    from src.intelligence.embeddings import embed_query
    from src.rag import retriever
    print(f"\n== Latency (median of {repeats} runs, after one warm-up)")
    ticker = next(iter(companies()))
    f = _wide_filters(ticker)

    def timed(fn) -> float:
        fn()
        samples = []
        for _ in range(repeats):
            t = time.perf_counter(); fn(); samples.append(time.perf_counter() - t)
        return statistics.median(samples) * 1000

    out: dict = {}
    with get_engine().connect() as conn:
        vector = embed_query("investor concerns about debt")
        for name, fn in (("SQL overview aggregate", lambda: metrics.overview(conn, f)),
                         ("SQL topic ranking", lambda: metrics.topics(conn, f)),
                         ("query embedding (CPU)", lambda: embed_query("investor concerns about debt")),
                         ("keyword search", lambda: retriever.keyword_search(conn, f, ["debt", "concerns"])),
                         ("vector search", lambda: retriever.vector_search(conn, f, vector))):
            out[name] = round(timed(fn), 1)
            print(f"   {name:28} {out[name]:8.1f} ms")
    print("   Database timings include the network round trip to the hosted database.")
    RESULTS_OUT["latency_ms"] = out


def _save() -> None:
    """Merge this run into results.json, so running one section keeps the others."""
    if not RESULTS_OUT:
        return
    existing = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    existing.update(RESULTS_OUT)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RESULTS.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "ingestion", "relevance", "sentiment",
                                            "retrieval", "latency", "all"])
    parser.add_argument("--per-company", type=int, default=40)
    args = parser.parse_args()
    if args.command == "export":
        return export(args.per_company)
    steps = {"ingestion": ingestion, "relevance": relevance, "sentiment": sentiment,
             "retrieval": retrieval, "latency": latency}
    for name, step in steps.items():
        if args.command in (name, "all"):
            try:
                step()
            except SystemExit as exc:
                print(f"\n== {name}: skipped. {exc}")
    _save()


if __name__ == "__main__":
    main()
