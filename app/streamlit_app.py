from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finsight.agent import LLMFinSightAgent, OpenAICompatibleActor  # noqa: E402
from finsight.audit import evaluate_result  # noqa: E402
from finsight.business_cases import load_business_cases  # noqa: E402
from finsight.training.teacher import TeacherError  # noqa: E402


RETRIEVAL_COMPARISON = PROJECT_ROOT / "artifacts" / "e2_e3_comparison_dev_100.json"
VALIDATION_GUIDE = PROJECT_ROOT / "docs" / "WEB_VALIDATION_GUIDE.md"
PROVIDER_PRESETS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "docs": "https://api-docs.deepseek.com/api/create-chat-completion",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": [],
        "docs": "https://platform.openai.com/docs/api-reference/chat/create",
    },
    "其他 OpenAI-compatible": {
        "base_url": "",
        "models": [],
        "docs": "",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">AI + 金融 · 企业经营分析 AGENT</div>
            <h1>FinSight</h1>
            <p>每一个财务结论，都能回到原始证据、受限公式与确定性验证。</p>
            <div class="chips">
              <span>Evidence-grounded</span><span>Safe DSL</span><span>Fail-closed</span>
            </div>
          </div>
          <div class="hero-score">
            <strong>Live LLM</strong>
            <span>唯一产品入口</span>
            <small>模型可配置 · 缺证会拒答</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline(result) -> None:
    stages = [
        ("01", "载入", "INGESTED"),
        ("02", "规划", "PLANNED"),
        ("03", "检索", "RETRIEVED"),
        ("04", "计算", "CALCULATED"),
        ("05", "验证", "VERIFIED"),
        ("06", "交付", "DELIVERED"),
    ]
    visited = {str(event.state) for event in result.trace}
    columns = st.columns(len(stages))
    for column, (index, label, state) in zip(columns, stages, strict=True):
        with column:
            completed = state in visited
            st.markdown(
                f"""
                <div class="stage {'stage-done' if completed else 'stage-idle'}">
                  <span>{index}</span><strong>{label}</strong>
                  <small>{'已完成' if completed else '未执行'}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if str(result.state) == "REFUSED":
        st.markdown(
            '<div class="refusal-strip">安全分支：关键证据不足 → REFUSED，不输出确定性结论</div>',
            unsafe_allow_html=True,
        )


def _render_evidence(task, result) -> None:
    retrieved_ids = {item.evidence.evidence_id for item in result.retrieved}
    required_ids = set(task.expected_evidence_ids)
    available_ids = {evidence.evidence_id for evidence in task.evidence}
    for evidence in task.evidence:
        retrieved = evidence.evidence_id in retrieved_ids
        required = evidence.evidence_id in required_ids
        with st.container(border=True):
            title_col, badge_col = st.columns([4, 1])
            with title_col:
                st.markdown(f"**{evidence.evidence_id}** · {evidence.source or '演示资料'}")
            with badge_col:
                if required and retrieved:
                    st.success("关键证据")
                elif retrieved:
                    st.info("已召回")
                elif required:
                    st.error("缺失")
                else:
                    st.caption("候选")
            st.write(evidence.text)
            st.caption(f"类型：{evidence.kind} · 页码：{evidence.page or '—'}")
    for missing_id in sorted(required_ids - available_ids):
        with st.container(border=True):
            st.error(f"缺失关键证据：{missing_id}")
            st.write("当前资料中没有该指标对应的可核验表格或文本，因此系统必须拒答。")


def _render_model_config() -> dict[str, Any]:
    st.markdown("### 连接模型")
    provider = st.selectbox(
        "模型服务商",
        list(PROVIDER_PRESETS),
        help="当前产品接入 Chat Completions 兼容协议；原生其他协议需通过独立适配器接入。",
    )
    preset = PROVIDER_PRESETS[provider]
    left, right = st.columns(2)
    with left:
        base_url = st.text_input(
            "Base URL",
            value=str(preset["base_url"]),
            key=f"base_url:{provider}",
            placeholder="https://provider.example/v1",
            help="可填写 API 根地址，也可直接填写以 /chat/completions 结尾的完整地址。",
        )
    with right:
        if preset["models"]:
            model_name = st.selectbox(
                "模型",
                preset["models"],
                key=f"model:{provider}",
            )
        else:
            model_name = st.text_input(
                "模型 ID",
                value="gpt-5.2" if provider == "OpenAI" else "",
                key=f"model:{provider}",
                placeholder="填写服务商公布的模型 ID",
            )

    api_key = st.text_input(
        "API Key",
        type="password",
        key=f"api_key:{provider}",
        placeholder="仅保存在当前 Streamlit 会话中",
    )
    st.caption(
        "密钥只用于从当前会话向所填 Base URL 发起请求，不写入配置文件、运行报告或下载结果。"
    )

    temperature: float | None = 0.0
    json_mode = True
    timeout_seconds = 60.0
    extra_body: dict[str, Any] = {}
    with st.expander("高级连接设置"):
        if provider == "DeepSeek":
            thinking_enabled = st.toggle(
                "启用 DeepSeek Thinking",
                value=False,
                help="根据 DeepSeek 官方 Chat Completions 参数发送 thinking.type。",
            )
            extra_body["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled"
            }
            if thinking_enabled:
                extra_body["reasoning_effort"] = "high"
        elif provider == "OpenAI":
            temperature = None
            st.caption("OpenAI 模式不主动发送 temperature，以兼容推理模型。")
        else:
            json_mode = st.toggle(
                "发送 response_format=json_object",
                value=True,
                help="若自定义服务不支持 JSON mode，可关闭；FinSight 仍会通过提示词和 Schema 校验动作。",
            )
            temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
        timeout_seconds = float(
            st.number_input("单次请求超时（秒）", 10, 180, 60, 10)
        )

    if preset["docs"]:
        st.markdown(f"接口参数参考：[查看 {provider} 官方文档]({preset['docs']})")

    return {
        "provider_name": provider,
        "base_url": base_url.strip(),
        "api_key": api_key.strip(),
        "model_name": str(model_name).strip(),
        "temperature": temperature,
        "json_mode": json_mode,
        "extra_body": extra_body,
        "timeout_seconds": timeout_seconds,
    }


def _current_result() -> tuple[Any | None, Any | None, dict[str, Any] | None]:
    return (
        st.session_state.get("agent_result"),
        st.session_state.get("agent_task"),
        st.session_state.get("agent_audit"),
    )


def _render_live_analysis() -> None:
    st.markdown("## 运行 FinSight Agent")
    st.caption(
        "问题输入 → LLM 生成搜索词 → 检索并选择证据 → LLM 生成公式 → 工具计算 → 验证 → 交付或拒答"
    )

    cases = load_business_cases()
    title_to_case = {case.title: case for case in cases}
    selected_title = st.selectbox(
        "选择分析任务",
        list(title_to_case),
        help="收入增长用于展示正常交付，存货周转率用于展示缺证拒答。",
    )
    task = title_to_case[selected_title]
    st.text_area(
        "样例输入（由所选资料场景载入）",
        value=task.question,
        height=80,
        disabled=True,
    )
    st.caption(
        "当前版本使用四组预解析样例资料验证 Agent 闭环，尚未开放任意财报上传或跨资料自由提问。"
    )
    with st.expander("查看本任务资料与边界"):
        for evidence in task.evidence:
            st.code(evidence.text, language="text")
        st.info(task.risk_note)

    config = _render_model_config()
    ready = bool(config["base_url"] and config["api_key"] and config["model_name"])
    run_agent = st.button(
        "运行 FinSight Agent",
        type="primary",
        width="stretch",
        disabled=not ready,
    )
    if not ready:
        st.info("选择服务商并填写 Base URL、模型 ID 和 API Key 后即可运行。")

    if run_agent:
        try:
            actor = OpenAICompatibleActor.from_connection(**config)
            with st.spinner(
                f"{config['provider_name']} / {config['model_name']} 正在自主检索、生成公式并接受验证……"
            ):
                result = LLMFinSightAgent(actor).run(task)
            st.session_state["agent_result"] = result
            st.session_state["agent_task"] = task
            st.session_state["agent_audit"] = evaluate_result(task, result)
        except (OSError, TeacherError) as exc:
            st.error(f"模型配置或接口调用失败：{exc}")

    result, result_task, audit = _current_result()
    if result is None or result_task is None or audit is None:
        return
    if result_task.task_id != task.task_id:
        st.info("当前显示的任务尚未运行；上一次结果可在“02 · 运行轨迹”中查看。")
        return

    st.markdown("### 本次分析结果")
    if result.verified:
        st.success("验证通过：数字、公式、证据与终止状态全部一致。")
    else:
        st.warning("安全拒答：关键输入不可核验，系统没有补造数字。")
    metadata = result.run_metadata
    columns = st.columns(4)
    columns[0].metric("模型服务", metadata.get("provider", "OpenAI-compatible"))
    columns[1].metric("模型", metadata.get("model", "—"))
    columns[2].metric("API 调用", int(metadata.get("api_calls", 0)))
    columns[3].metric(
        "端到端耗时", f"{float(metadata.get('elapsed_seconds', 0.0)):.2f}s"
    )
    with st.container(border=True):
        st.markdown("#### 经营分析报告")
        st.markdown(result.report_markdown)
    st.caption("完整动作、搜索词、证据选择、公式和验证回执请查看“02 · 运行轨迹”。")


def _render_run_trace() -> None:
    st.markdown("## 运行轨迹")
    result, task, audit = _current_result()
    if result is None or task is None or audit is None:
        st.info("请先在“01 · 现场分析”中运行一次 FinSight Agent。")
        return

    _render_pipeline(result)
    metadata = result.run_metadata
    identity_columns = st.columns(4)
    identity_columns[0].metric("服务商", metadata.get("provider", "—"))
    identity_columns[1].metric("模型", metadata.get("model", "—"))
    identity_columns[2].metric("环境动作", metadata.get("environment_steps", 0))
    identity_columns[3].metric("Schema 修复", metadata.get("schema_repairs", 0))
    st.caption(
        f"Prompt={metadata.get('prompt_version')} · "
        f"termination={metadata.get('termination_reason')} · hidden_gold_exposed=false"
    )

    action_tab, evidence_tab, verify_tab = st.tabs(["动作与公式", "证据", "验证结果"])
    with action_tab:
        st.markdown("### LLM 与工具交互")
        for event in result.trace:
            icon = "✅" if str(event.status) == "completed" else "⛔"
            with st.expander(
                f"{icon} {event.event_id} · {event.action or event.state} · {event.message}",
                expanded=event.action in {"search", "emit_program", "verify", "deliver", "abstain"},
            ):
                arguments = event.details.get("arguments") if event.details else None
                observation = event.details.get("observation") if event.details else None
                if arguments is not None:
                    st.caption("动作参数")
                    st.json(arguments)
                if observation is not None:
                    st.caption("环境回执")
                    st.json(observation)
                elif event.details:
                    st.json(event.details)
        st.markdown("### 模型生成的受限公式")
        st.code(result.formula or "未生成公式", language="text")
    with evidence_tab:
        _render_evidence(task, result)
    with verify_tab:
        outcome = audit["outcome_panel"]
        columns = st.columns(4)
        columns[0].metric("终局", str(result.state))
        columns[1].metric(
            "结论", f"{result.answer:.2f}{result.unit}" if result.answer is not None else "拒答"
        )
        columns[2].metric("证据覆盖", _percent(outcome["evidence_coverage"]))
        columns[3].metric(
            "程序安全",
            "通过" if audit["deterministic_panel"]["program_safe"] else "未通过",
        )
        st.write(outcome["reason"])

    st.download_button(
        "下载本次运行报告 JSON",
        json.dumps(
            {"result": result.to_dict(), "evaluation": audit},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        file_name=f"finsight-{result.task_id}-run.json",
        mime="application/json",
        width="stretch",
    )


def _render_comparison_metric(label: str, metric: dict[str, float]) -> None:
    baseline = float(metric["baseline"])
    candidate = float(metric["candidate"])
    delta = float(metric["delta"])
    with st.container(border=True):
        title_col, value_col = st.columns([3, 2])
        with title_col:
            st.markdown(f"**{label}**")
            st.caption(f"基础 BM25 {_percent(baseline)}")
            st.progress(min(max(baseline, 0.0), 1.0))
            st.caption(f"当前 Table-aware BM25 {_percent(candidate)}")
            st.progress(min(max(candidate, 0.0), 1.0))
        with value_col:
            st.metric("当前检索器", _percent(candidate), f"{delta * 100:+.2f} pp")


def _render_evaluation() -> None:
    st.markdown("## 效果评测")
    result, _task, audit = _current_result()
    st.markdown("### 实时 Agent 本次运行")
    if result is None or audit is None:
        st.info("运行 Agent 后，这里会显示本次模型、终局、证据覆盖和轨迹状态。")
    else:
        outcome = audit["outcome_panel"]
        columns = st.columns(4)
        columns[0].metric("模型", result.run_metadata.get("model", "—"))
        columns[1].metric("终局", str(result.state))
        columns[2].metric("证据覆盖", _percent(outcome["evidence_coverage"]))
        columns[3].metric(
            "轨迹状态",
            "通过" if audit["trajectory_panel"]["state_path_valid"] else "异常",
        )

    comparison = _read_json(RETRIEVAL_COMPARISON)
    if comparison:
        st.markdown("### 检索指标")
        st.caption(
            f"FinQA dev 前 {comparison['fixed_denominator']} 条、Top-{comparison['top_k']}；这里只评价证据检索。"
        )
        left, right = st.columns(2)
        with left:
            _render_comparison_metric(
                "Top-5 证据召回率（Recall@5）",
                comparison["metrics"]["recall_at_k"],
            )
            _render_comparison_metric(
                "完整证据覆盖率",
                comparison["metrics"]["gold_evidence_coverage"],
            )
        with right:
            _render_comparison_metric(
                "首条正确证据排名（MRR）",
                comparison["metrics"]["mrr"],
            )
            _render_comparison_metric(
                "表格证据召回率",
                comparison["metrics"]["table_evidence_recall"],
            )
        text_delta = comparison["metrics"]["text_evidence_recall"]["delta"]
        st.warning(
            "Table-aware 提升了表格证据排序，但文本证据召回变化为 "
            f"{text_delta * 100:+.2f} pp；后续用混合检索和 Reranker 修复。"
        )

def _render_project_info() -> None:
    st.markdown("## 项目说明")
    st.markdown("### 核心流程")
    st.code(
        "问题输入 → LLM 生成搜索词 → 检索证据 → 打开并选择证据 → "
        "LLM 生成公式 → 工具计算 → 验证 → 交付或拒答",
        language="text",
    )

    model_col, tool_col = st.columns(2)
    with model_col:
        st.markdown("### 模型与接口")
        st.markdown(
            """
            - 主入口始终运行实时 LLM Agent。
            - 支持 DeepSeek、OpenAI 与其他 OpenAI-compatible Chat Completions 服务。
            - 用户填写服务商、Base URL、模型 ID 和 API Key。
            - API Key 仅存在当前会话内存，不进入运行报告或本地文件。
            - 模型只生成结构化动作，不能绕过 Action Guard 和 Verifier。
            """
        )
        st.markdown(
            "[DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion) · "
            "[OpenAI Chat Completions 文档](https://platform.openai.com/docs/api-reference/chat/create)"
        )
    with tool_col:
        st.markdown("### 工具与 Agent 能力")
        st.markdown(
            """
            - `search`：LLM 生成查询词并调用 Table-aware BM25。
            - `open_evidence` / `select_evidence`：显式查看并选择依据。
            - `emit_program`：LLM 生成 FinQA 风格受限计算 DSL。
            - `calculate`：确定性工具执行，不运行任意 Python。
            - `verify`：检查数字、公式、证据覆盖和状态顺序。
            - `deliver` / `abstain`：验证通过后交付，缺证时拒答。
            """
        )

    data_col, compliance_col = st.columns(2)
    with data_col:
        st.markdown("### 数据")
        st.markdown(
            """
            - FinQA 官方公开数据：用于检索、数值推理和执行评测。
            - 合成非敏感业务案例：用于稳定展示正常交付与缺证拒答。
            - 当前版本使用预解析资料；真实 PDF/OCR 和中文财报属于下一阶段。
            - 表格处理保留行名、年份、单位、正负号和 evidence_id。
            """
        )
    with compliance_col:
        st.markdown("### 合规边界")
        st.markdown(
            """
            - 仅供企业经营分析辅助。
            - 不构成投资建议、授信审批、审计结论或自动风控决策。
            - 证据缺失、计算失败或验证不通过时关闭式停止。
            - 未来接入真实财报时增加授权、脱敏、访问控制和删除机制。
            """
        )

    st.markdown("### 后续落地")
    st.markdown(
        "真实 PDF/OCR 与中文财报 → 混合检索与 Reranker → Base LLM/Agent 同题对照 → "
        "Action-only SFT 与 GRPO → 容器化复现 → 真实用户小规模验证"
    )

    if VALIDATION_GUIDE.exists():
        st.download_button(
            "下载网页验证指南",
            data=VALIDATION_GUIDE.read_text(encoding="utf-8"),
            file_name="FinSight_网页验证指南.md",
            mime="text/markdown",
            width="stretch",
        )


st.set_page_config(
    page_title="FinSight · 可验证财报分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --navy:#0b1728; --ink:#172033; --muted:#687386; --green:#1f9d74; --mint:#dff7ed; }
      .stApp { background: linear-gradient(180deg, #f4f7f9 0, #ffffff 360px); color: var(--ink); }
      .block-container { max-width: 1280px; padding-top: 1.6rem; padding-bottom: 4rem; }
      .hero { display:flex; justify-content:space-between; gap:2rem; align-items:center; padding:2.3rem 2.5rem;
              border-radius:24px; color:white; background:radial-gradient(circle at 80% 10%, #1d7668 0, #102b3b 34%, #091523 72%);
              box-shadow:0 22px 60px rgba(9,21,35,.18); margin-bottom:1.2rem; }
      .hero h1 { font-size:3.3rem; margin:.15rem 0; letter-spacing:-.05em; }
      .hero p { color:#d6e5e4; font-size:1.05rem; margin:.3rem 0 1.1rem; }
      .eyebrow { color:#71e0bd; font-size:.78rem; font-weight:750; letter-spacing:.12em; }
      .chips { display:flex; flex-wrap:wrap; gap:.5rem; }
      .chips span { border:1px solid rgba(255,255,255,.23); border-radius:999px; padding:.32rem .72rem; font-size:.75rem; color:#e7f5f1; }
      .hero-score { min-width:225px; padding:1.2rem 1.4rem; border:1px solid rgba(255,255,255,.18); border-radius:18px; background:rgba(255,255,255,.08); }
      .hero-score strong, .hero-score span, .hero-score small { display:block; }
      .hero-score strong { color:#77e6c3; font-size:2.1rem; }
      .hero-score span { font-weight:700; }
      .hero-score small { color:#bdd0d0; margin-top:.3rem; }
      .stage { min-height:92px; border-radius:14px; padding:.8rem; border:1px solid #dce4ea; background:white; }
      .stage span, .stage strong, .stage small { display:block; }
      .stage span { font-size:.7rem; color:#8290a3; }
      .stage strong { margin:.15rem 0; }
      .stage small { color:#788396; }
      .stage-done { border-color:#8bd8bf; background:linear-gradient(180deg,#f3fcf8,#e5f8f0); }
      .stage-done strong { color:#16795a; }
      .stage-idle { opacity:.55; }
      .refusal-strip { margin-top:.55rem; padding:.65rem 1rem; border-radius:10px; background:#fff1e8; color:#9b4e18; border:1px solid #f0c6a8; }
      [data-testid="stMetric"] { background:white; border:1px solid #e1e7ec; padding:.9rem 1rem; border-radius:14px; }
      [data-testid="stSidebar"] { background:#0d1b2b; }
      [data-testid="stSidebar"] * { color:#eef5f5; }
      [data-testid="stSidebar"] .stAlert * { color:inherit; }
      [data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
      .stButton > button[kind="primary"] { background:#168765; border-color:#168765; }
      div[data-testid="stExpander"] { background:white; border-radius:12px; }
      @media (max-width: 760px) {
        .hero { display:block; padding:1.6rem; }
        .hero h1 { font-size:2.5rem; }
        .hero-score { margin-top:1.2rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## FinSight")
    st.caption("实时财报经营分析 Agent")
    st.success("产品入口：实时 LLM Agent")
    st.markdown("**核心能力**")
    st.write("模型自主生成搜索词")
    st.write("证据可定位")
    st.write("公式可复算")
    st.write("轨迹可回放")
    st.write("证据不足会拒答")
    result, _task, _audit = _current_result()
    if result is not None:
        st.divider()
        st.caption("最近一次运行")
        st.write(result.run_metadata.get("provider", "OpenAI-compatible"))
        st.write(result.run_metadata.get("model", "—"))
        st.write(str(result.state))
    st.divider()
    st.caption("API Key 仅保存在当前会话，不写入文件和报告。")
    st.caption("系统仅辅助经营分析，不构成投资、授信、审计或风控决策。")

_render_header()
analysis_tab, trace_tab, evaluation_tab, project_tab = st.tabs(
    ["01 · 现场分析", "02 · 运行轨迹", "03 · 效果评测", "04 · 项目说明"]
)
with analysis_tab:
    _render_live_analysis()
with trace_tab:
    _render_run_trace()
with evaluation_tab:
    _render_evaluation()
with project_tab:
    _render_project_info()

st.divider()
st.caption("FinSight · Evidence-grounded financial analysis agent · 缺少关键证据时拒绝交付")
