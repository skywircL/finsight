from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from finsight.data import load_finqa
from finsight.training.collection import collect_teacher_dataset
from finsight.training.splits import (
    DEFAULT_TRAIN_COUNTS,
    build_split_artifacts,
    file_sha256,
    load_partition_ids,
)
from finsight.training.sft import build_sft_dataset
from finsight.training.trajectories import build_oracle_dataset
from finsight.training.teacher import OpenAICompatibleTeacher, scripted_oracle_teacher
from finsight.training.teacher_audit import audit_teacher_output


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _build_splits(args: argparse.Namespace) -> None:
    manifest = build_split_artifacts(
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        output_dir=args.output_dir,
        seed=args.seed,
        train_counts={
            "sft_train": args.sft_train,
            "sft_validation": args.sft_validation,
            "grpo_train": args.grpo_train,
            "grpo_validation": args.grpo_validation,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _build_oracle(args: argparse.Namespace) -> None:
    allowed = {"sft_train", "sft_validation"}
    if args.partition not in allowed:
        raise ValueError("Oracle SFT 构建只允许使用 sft_train 或 sft_validation。")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition_ids = load_partition_ids(manifest_path, args.partition)
    metadata = manifest["partitions"][args.partition]
    source = manifest["sources"][metadata["source_split"]]
    source_path = Path(source["path"])
    if file_sha256(source_path) != source["sha256"]:
        raise ValueError("原始 FinQA 文件 SHA-256 与冻结 manifest 不一致。")
    sample_index = {sample.sample_id: sample for sample in load_finqa(source_path)}
    missing = set(partition_ids) - set(sample_index)
    if missing:
        raise ValueError(f"冻结划分中有 {len(missing)} 个 task_id 无法从原始数据恢复。")
    ordered_samples = [sample_index[task_id] for task_id in partition_ids]
    rows, audits, summary = build_oracle_dataset(ordered_samples, limit=args.limit)

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    accepted_path = destination / "accepted.jsonl"
    audit_path = destination / "audit.jsonl"
    _write_jsonl(accepted_path, rows)
    _write_jsonl(audit_path, audits)
    summary.update(
        {
            "partition": args.partition,
            "partition_count": len(partition_ids),
            "partition_sha256": metadata["sha256"],
            "source_sha256": source["sha256"],
            "outputs": {
                "accepted": {
                    "file": accepted_path.name,
                    "rows": len(rows),
                    "sha256": file_sha256(accepted_path),
                },
                "audit": {
                    "file": audit_path.name,
                    "rows": len(audits),
                    "sha256": file_sha256(audit_path),
                },
            },
        }
    )
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _partition_samples(manifest_path: str | Path, partition: str):
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    partition_ids = load_partition_ids(manifest_file, partition)
    metadata = manifest["partitions"][partition]
    source = manifest["sources"][metadata["source_split"]]
    source_path = Path(source["path"])
    if file_sha256(source_path) != source["sha256"]:
        raise ValueError("原始 FinQA 文件 SHA-256 与冻结 manifest 不一致。")
    sample_index = {sample.sample_id: sample for sample in load_finqa(source_path)}
    missing = set(partition_ids) - set(sample_index)
    if missing:
        raise ValueError(f"冻结划分中有 {len(missing)} 个 task_id 无法从原始数据恢复。")
    return [sample_index[task_id] for task_id in partition_ids]


def _collect_teacher(args: argparse.Namespace) -> None:
    if args.partition not in {"sft_train", "sft_validation"}:
        raise ValueError("教师采集只允许使用 sft_train 或 sft_validation。")
    samples = _partition_samples(args.manifest, args.partition)
    if args.provider == "scripted":
        teacher_factory = scripted_oracle_teacher
    else:
        def teacher_factory(_sample):
            if args.config_file:
                return OpenAICompatibleTeacher.from_config_file(args.config_file)
            return OpenAICompatibleTeacher.from_env()
    summary = collect_teacher_dataset(
        samples,
        teacher_factory,
        output_dir=args.output_dir,
        limit=args.limit,
        max_steps=args.max_steps,
    )
    summary["partition"] = args.partition
    summary["provider"] = args.provider
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _build_sft_data(args: argparse.Namespace) -> None:
    manifest = build_sft_dataset(
        args.train_source,
        args.validation_source,
        output_dir=args.output_dir,
        max_context_tokens=args.max_context_tokens,
        allow_oracle=args.allow_oracle,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _audit_teacher(args: argparse.Namespace) -> None:
    samples = _partition_samples(args.manifest, args.partition)
    report = audit_teacher_output(args.output_dir, samples)
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare frozen FinSight training data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    splits = subparsers.add_parser("splits", help="freeze task-disjoint FinQA partitions")
    splits.add_argument("--train", default="data/raw/finqa/train.json")
    splits.add_argument("--dev", default="data/raw/finqa/dev.json")
    splits.add_argument("--test", default="data/raw/finqa/test.json")
    splits.add_argument("--output-dir", default="data/splits/finqa_v1")
    splits.add_argument("--seed", default="finsight-finqa-v1")
    splits.add_argument("--sft-train", type=int, default=DEFAULT_TRAIN_COUNTS["sft_train"])
    splits.add_argument(
        "--sft-validation", type=int, default=DEFAULT_TRAIN_COUNTS["sft_validation"]
    )
    splits.add_argument("--grpo-train", type=int, default=DEFAULT_TRAIN_COUNTS["grpo_train"])
    splits.add_argument(
        "--grpo-validation", type=int, default=DEFAULT_TRAIN_COUNTS["grpo_validation"]
    )
    splits.set_defaults(handler=_build_splits)

    oracle = subparsers.add_parser("oracle", help="replay and audit Oracle SFT trajectories")
    oracle.add_argument("--manifest", default="data/splits/finqa_v1/manifest.json")
    oracle.add_argument("--partition", choices=["sft_train", "sft_validation"], required=True)
    oracle.add_argument("--output-dir", required=True)
    oracle.add_argument("--limit", type=int)
    oracle.set_defaults(handler=_build_oracle)

    teacher = subparsers.add_parser(
        "teacher", help="collect resumable teacher trajectories through the environment"
    )
    teacher.add_argument("--manifest", default="data/splits/finqa_v1/manifest.json")
    teacher.add_argument("--partition", choices=["sft_train", "sft_validation"], required=True)
    teacher.add_argument("--output-dir", required=True)
    teacher.add_argument("--provider", choices=["scripted", "openai"], default="scripted")
    teacher.add_argument(
        "--limit",
        type=int,
        help="本次新增尝试数；断点续跑时会跳过 raw.jsonl 中已有 task_id",
    )
    teacher.add_argument("--max-steps", type=int)
    teacher.add_argument(
        "--config-file",
        help="本地 KEY=VALUE 私密配置文件；不要提交到版本控制",
    )
    teacher.set_defaults(handler=_collect_teacher)

    sft_data = subparsers.add_parser(
        "sft-data", help="validate and freeze Action-only SFT train/validation rows"
    )
    sft_data.add_argument("--train-source", required=True)
    sft_data.add_argument("--validation-source", required=True)
    sft_data.add_argument("--output-dir", required=True)
    sft_data.add_argument("--max-context-tokens", type=int, default=24_576)
    sft_data.add_argument(
        "--allow-oracle",
        action="store_true",
        help="仅用于生成 Oracle 管线冒烟数据，不代表正式教师数据",
    )
    sft_data.set_defaults(handler=_build_sft_data)

    audit = subparsers.add_parser(
        "audit-teacher", help="build deterministic quality audit for teacher raw.jsonl"
    )
    audit.add_argument("--manifest", default="data/splits/finqa_v1/manifest.json")
    audit.add_argument("--partition", choices=["sft_train", "sft_validation"], required=True)
    audit.add_argument("--output-dir", required=True)
    audit.set_defaults(handler=_audit_teacher)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
