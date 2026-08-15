from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean
from math import isclose

from finsight.agent import FinSightAgent
from finsight.audit import evaluate_result
from finsight.business_cases import load_business_cases
from finsight.data import load_finqa
from finsight.retrieval import BM25Retriever, TableAwareBM25Retriever
from finsight.tools import CalculationError, execute_program


def _retriever(method: str, evidences):
    if method == "bm25":
        return BM25Retriever(evidences)
    if method == "table_aware":
        return TableAwareBM25Retriever(evidences)
    raise ValueError(f"Unsupported retrieval method: {method}")


def evaluate_retrieval(
    path: str | Path,
    *,
    limit: int = 100,
    top_k: int = 5,
    method: str = "bm25",
) -> dict:
    samples = load_finqa(path, limit=limit)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    exact_coverage = 0
    kind_hits = {"table": 0, "text": 0}
    kind_totals = {"table": 0, "text": 0}

    for sample in samples:
        ranked = _retriever(method, sample.evidences).search(sample.question, top_k=top_k)
        retrieved_ids = [item.evidence.evidence_id for item in ranked]
        gold = set(sample.gold_evidence_ids)
        hits = gold & set(retrieved_ids)
        recalls.append(len(hits) / len(gold) if gold else 1.0)
        exact_coverage += int(gold <= set(retrieved_ids))
        hit_ranks = [index + 1 for index, item in enumerate(retrieved_ids) if item in gold]
        reciprocal_ranks.append(1 / min(hit_ranks) if hit_ranks else 0.0)
        for evidence_id in gold:
            kind = "table" if evidence_id.startswith("table_") else "text"
            kind_totals[kind] += 1
            kind_hits[kind] += int(evidence_id in retrieved_ids)

    return {
        "dataset": str(path),
        "retrieval_method": method,
        "samples": len(samples),
        "top_k": top_k,
        "recall_at_k": mean(recalls) if recalls else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "gold_evidence_coverage": exact_coverage / len(samples) if samples else 0.0,
        "table_evidence_recall": kind_hits["table"] / kind_totals["table"] if kind_totals["table"] else 0.0,
        "text_evidence_recall": kind_hits["text"] / kind_totals["text"] if kind_totals["text"] else 0.0,
    }


def retrieval_failures(
    path: str | Path,
    *,
    limit: int = 100,
    top_k: int = 5,
    method: str = "bm25",
) -> list[dict]:
    failures = []
    for sample in load_finqa(path, limit=limit):
        ranked = _retriever(method, sample.evidences).search(sample.question, top_k=top_k)
        retrieved_ids = [item.evidence.evidence_id for item in ranked]
        missing = sorted(set(sample.gold_evidence_ids) - set(retrieved_ids))
        if missing:
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "gold_evidence_ids": list(sample.gold_evidence_ids),
                    "retrieved_evidence_ids": retrieved_ids,
                    "missing_evidence_ids": missing,
                    "top_evidence": [
                        {
                            "id": item.evidence.evidence_id,
                            "kind": item.evidence.kind,
                            "score": item.score,
                            "text": item.evidence.text,
                        }
                        for item in ranked[:3]
                    ],
                }
            )
    return failures


def evaluate_gold_programs(path: str | Path, *, limit: int | None = None) -> dict:
    samples = load_finqa(path, limit=limit)
    matches = 0
    errors = 0
    for sample in samples:
        try:
            predicted = execute_program(sample.program, table=sample.table).value
            if isinstance(predicted, str):
                matches += int(predicted.lower() == str(sample.answer).lower())
            else:
                matches += int(
                    isclose(round(predicted, 5), float(sample.answer), rel_tol=1e-3, abs_tol=1e-3)
                )
        except (CalculationError, TypeError, ValueError):
            errors += 1
    return {
        "dataset": str(path),
        "samples": len(samples),
        "executed": len(samples) - errors,
        "execution_coverage": (len(samples) - errors) / len(samples) if samples else 0.0,
        "gold_program_match_rate": matches / len(samples) if samples else 0.0,
        "errors": errors,
    }


def evaluate_business_suite(path: str | Path) -> dict:
    """Evaluate the frozen business suite with fixed-denominator panels."""

    source = Path(path)
    evaluations = []
    for task in load_business_cases(source):
        result = FinSightAgent().run(task)
        evaluations.append(evaluate_result(task, result))

    reward_values = [item["outcome_panel"]["reward"] for item in evaluations]
    delivered = sum(
        item["outcome_panel"]["terminal_state"] == "DELIVERED" for item in evaluations
    )
    correct_refusals = sum(
        item["outcome_panel"]["outcome"] == "correct_refusal" for item in evaluations
    )
    valid = sum(item["outcome_panel"]["reward_valid"] for item in evaluations)
    return {
        "schema_version": "finsight-business-suite-v1",
        "suite": str(source),
        "suite_sha256": sha256(source.read_bytes()).hexdigest(),
        "fixed_denominator": len(evaluations),
        "summary": {
            "delivered": delivered,
            "correct_refusals": correct_refusals,
            "reward_valid": valid,
            "mean_reward": mean(reward_values) if reward_values else 0.0,
        },
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the FinSight BM25 baseline")
    parser.add_argument("--data", default="data/raw/finqa/dev.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=["bm25", "table_aware"], default="bm25")
    parser.add_argument("--output", default="artifacts/bm25_baseline.json")
    parser.add_argument("--failures-output", default="artifacts/bm25_failures.json")
    parser.add_argument("--program-output", default="artifacts/gold_program_check.json")
    parser.add_argument("--business-data", default="data/business_eval/cases.json")
    parser.add_argument("--business-output", default="artifacts/business_evaluation_v1.json")
    args = parser.parse_args()
    result = evaluate_retrieval(
        args.data,
        limit=args.limit,
        top_k=args.top_k,
        method=args.retriever,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = retrieval_failures(
        args.data,
        limit=args.limit,
        top_k=args.top_k,
        method=args.retriever,
    )
    failures_output = Path(args.failures_output)
    failures_output.parent.mkdir(parents=True, exist_ok=True)
    failures_output.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    program_result = evaluate_gold_programs(args.data, limit=args.limit)
    program_output = Path(args.program_output)
    program_output.parent.mkdir(parents=True, exist_ok=True)
    program_output.write_text(
        json.dumps(program_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    business_result = evaluate_business_suite(args.business_data)
    business_output = Path(args.business_output)
    business_output.parent.mkdir(parents=True, exist_ok=True)
    business_output.write_text(
        json.dumps(business_result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
