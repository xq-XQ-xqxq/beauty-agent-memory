import json
from types import SimpleNamespace

from label_engine import ServiceInsightGenerator


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload, ensure_ascii=False)
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self, payload):
        self.completions = FakeCompletions(payload)
        self.chat = SimpleNamespace(completions=self.completions)


def test_second_mapping_accepts_current_message_and_memory_sources():
    client = FakeClient(
        {
            "service_focus": "用户反馈面膜使用后泛红瘙痒",
            "labels": [
                {
                    "title": "使用后泛红瘙痒",
                    "content": "用户使用面膜后出现泛红和瘙痒",
                    "priority": "urgent",
                    "source_ids": ["msg_1", "m1"],
                },
                {
                    "title": "无依据标签",
                    "content": "不应通过",
                    "priority": "high",
                    "source_ids": ["unknown"],
                },
            ],
            "service_suggestions": [
                {
                    "action": "safety_attention",
                    "content": "先确认当前是否还有其他不适",
                    "basis_ids": ["msg_1"],
                }
            ],
        }
    )
    generator = ServiceInsightGenerator(client=client)

    insight = generator.generate(
        [{"role": "user", "content": "用了面膜后泛红发痒"}],
        [
            {
                "id": "m1",
                "memory": "用户是敏感肌",
                "metadata": {"memory_key": "body.skin_sensitivity"},
            }
        ],
    )

    assert insight.service_focus == "用户反馈面膜使用后泛红瘙痒"
    assert len(insight.labels) == 1
    assert insight.labels[0].source_ids == ("msg_1", "m1")
    assert len(insight.service_suggestions) == 1
    assert client.completions.kwargs["temperature"] == 0


def test_second_mapping_limits_sorts_and_validates_output():
    labels = [
        {
            "title": f"标签{i}",
            "content": f"内容{i}",
            "priority": "urgent" if i == 5 else "normal",
            "source_ids": ["msg_1"],
        }
        for i in range(7)
    ]
    suggestions = [
        {"action": "clarify", "content": f"建议{i}", "basis_ids": ["msg_1"]}
        for i in range(3)
    ]
    suggestions.append(
        {"action": "approve_refund", "content": "非法动作", "basis_ids": ["msg_1"]}
    )
    generator = ServiceInsightGenerator(
        client=FakeClient(
            {
                "service_focus": "测试",
                "labels": labels,
                "service_suggestions": suggestions,
            }
        )
    )

    insight = generator.generate([{"role": "user", "content": "测试消息"}], [])

    assert len(insight.labels) == 5
    assert insight.labels[0].title == "标签5"
    assert len(insight.service_suggestions) == 2


def test_second_mapping_works_without_long_term_memory():
    generator = ServiceInsightGenerator(
        client=FakeClient(
            {
                "service_focus": "用户当前很生气",
                "labels": [
                    {
                        "title": "当前情绪强烈",
                        "content": "用户明确表示现在很生气",
                        "priority": "urgent",
                        "source_ids": ["msg_1"],
                    }
                ],
                "service_suggestions": [],
            }
        )
    )

    insight = generator.generate([{"role": "user", "content": "我现在很生气"}], [])

    assert insight.labels[0].title == "当前情绪强烈"
    assert insight.labels[0].source_ids == ("msg_1",)


def test_second_mapping_filters_inference_and_direct_disposal_advice():
    generator = ServiceInsightGenerator(
        client=FakeClient(
            {
                "service_focus": "用户使用后刺痛",
                "labels": [
                    {
                        "title": "敏感肌",
                        "content": "用户是敏感肌，可能耐受度较低",
                        "priority": "high",
                        "source_ids": ["m1"],
                    }
                ],
                "service_suggestions": [
                    {
                        "action": "safety_attention",
                        "content": "建议立即停用并及时就医",
                        "basis_ids": ["msg_1"],
                    }
                ],
            }
        )
    )

    insight = generator.generate(
        [{"role": "user", "content": "用了以后刺痛"}],
        [{"id": "m1", "memory": "用户是敏感肌", "metadata": {}}],
    )

    assert insight.labels == ()
    assert insight.service_suggestions == ()
