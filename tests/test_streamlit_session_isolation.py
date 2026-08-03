from pathlib import Path

from streamlit.testing.v1 import AppTest


HERE = Path(__file__).resolve()
APP_FILE = HERE.with_name("streamlit_app.py")
if not APP_FILE.exists():
    APP_FILE = HERE.parents[1] / "streamlit_app.py"


def button(at, label):
    return next(item for item in at.button if item.label == label)


def test_two_users_keep_independent_conversation_state():
    at = AppTest.from_file(str(APP_FILE)).run(timeout=20)

    at.text_input[-1].set_value("买家一的消息")
    button(at, "发送").click().run(timeout=20)
    buyer_one_id = at.session_state["active_conversation_id"]

    button(at, "新建").click().run(timeout=20)
    next(item for item in at.text_input if item.label == "用户 ID").set_value("buyer_2")
    button(at, "创建并进入").click().run(timeout=20)
    buyer_two_id = at.session_state["active_conversation_id"]

    at.text_input[-1].set_value("买家二的消息")
    button(at, "发送").click().run(timeout=20)

    sessions = at.session_state["sessions"]
    assert buyer_one_id != buyer_two_id
    assert sessions[buyer_one_id]["user_id"] == "demo_customer_001"
    assert sessions[buyer_one_id]["messages"] == [
        {"role": "user", "content": "买家一的消息"}
    ]
    assert sessions[buyer_two_id]["user_id"] == "buyer_2"
    assert sessions[buyer_two_id]["messages"] == [
        {"role": "user", "content": "买家二的消息"}
    ]
