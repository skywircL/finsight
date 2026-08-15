from streamlit.testing.v1 import AppTest


def test_dashboard_exposes_live_agent_as_the_only_product_entry():
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "01 · 现场分析",
        "02 · 运行轨迹",
        "03 · 效果评测",
        "04 · 项目说明",
    ]
    assert not app.radio
    assert app.selectbox[0].label == "选择分析任务"
    assert app.selectbox[0].value == "收入增长分析"
    assert app.selectbox[1].label == "模型服务商"
    assert app.selectbox[1].value == "DeepSeek"
    assert app.selectbox[2].label == "模型"
    assert app.selectbox[2].value == "deepseek-v4-flash"
    assert app.button[0].label == "运行 FinSight Agent"
    assert app.button[0].disabled
    assert not any("确定性工作流基线" in item.label for item in app.expander)
    assert not any(metric.label == "终局" for metric in app.metric)


def test_dashboard_allows_a_custom_openai_compatible_model():
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=30)
    app.selectbox[1].select("其他 OpenAI-compatible").run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.text_input] == [
        "Base URL",
        "模型 ID",
        "API Key",
    ]
    assert app.button[0].disabled

    app.selectbox[0].select("存货周转率核查").run(timeout=30)
    assert "inventory turnover" in app.text_area[0].value
