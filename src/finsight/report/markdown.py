from __future__ import annotations

from finsight.models import AnalysisTask, RetrievedEvidence


DISCLAIMER = "仅供经营分析辅助，不构成投资、授信、审计或风控决策。"


def build_report(
    task: AnalysisTask,
    *,
    answer: float | None,
    formula: str | None,
    retrieved: list[RetrievedEvidence],
    verified: bool,
    refusal_reason: str | None = None,
) -> str:
    if not verified:
        return (
            f"## {task.title}\n\n"
            f"**结论：证据不足，暂不输出确定性判断。**\n\n"
            f"原因：{refusal_reason or '验证未通过。'}\n\n"
            f"> {DISCLAIMER}"
        )

    evidence_lines = "\n".join(
        f"- `{item.evidence.evidence_id}`：{item.evidence.text}（BM25={item.score:.3f}）"
        for item in retrieved
        if item.score > 0
    )
    risk = task.risk_note or "该指标变化需要结合行业、基数和其他财务科目进一步核查。"
    return (
        f"## {task.title}\n\n"
        f"**结论：{answer:.2f}{task.unit}**\n\n"
        f"- 分析问题：{task.question}\n"
        f"- 计算程序：`{formula}`\n"
        f"- 验证状态：已通过数字、公式与证据覆盖检查\n"
        f"- 风险线索：{risk}\n\n"
        f"### 原始依据\n\n{evidence_lines}\n\n"
        f"> {DISCLAIMER}"
    )

