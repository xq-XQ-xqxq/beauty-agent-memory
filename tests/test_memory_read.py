from memory_service import CustomerMemoryService


class FakeMemory:
    def __init__(self):
        self.calls = []

    def get_all(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "id": "m2",
                    "user_id": "u1",
                    "memory": "用户喜欢无香产品",
                    "metadata": {"memory_key": "preference.product_attribute"},
                },
                {
                    "id": "m1",
                    "user_id": "u1",
                    "memory": "用户是敏感肌",
                    "metadata": {"memory_key": "body.skin_sensitivity"},
                },
                {
                    "id": "foreign",
                    "user_id": "u2",
                    "memory": "另一个用户是油性皮肤",
                    "metadata": {"memory_key": "body.skin_type"},
                },
            ]
        }


class FakeStore:
    def get(self, *args):
        return None

    def record(self, *args):
        return None


def test_get_user_memories_reads_all_without_vector_search():
    memory = FakeMemory()
    service = CustomerMemoryService(memory=memory, processed_turn_store=FakeStore())

    result = service.get_user_memories("u1")

    assert memory.calls == [{"filters": {"user_id": "u1"}, "top_k": 100}]
    assert [item["id"] for item in result] == ["m1", "m2"]


def test_get_user_memories_rejects_foreign_and_unscoped_records():
    class LeakyMemory:
        def get_all(self, **kwargs):
            return {
                "results": [
                    {"id": "mine", "user_id": "buyer_1", "memory": "我的记忆"},
                    {"id": "foreign", "user_id": "buyer_2", "memory": "别人的记忆"},
                    {"id": "missing", "memory": "没有归属的旧数据"},
                ]
            }

    service = CustomerMemoryService(memory=LeakyMemory(), processed_turn_store=FakeStore())

    result = service.get_user_memories("buyer_1")

    assert [item["id"] for item in result] == ["mine"]
