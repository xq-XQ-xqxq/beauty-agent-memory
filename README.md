# 美妆电商客服 AI Agent（Profile+Event 双通道记忆系统）

面向美妆电商人工客服的 AI 辅助工作台：维护用户长期记忆，在客服分析时由 Agent 主动调用工具取证（查知识库 / 查画像 / 查事件 / 查成分安全），生成带证据引用的服务焦点、动态标签与服务建议，售后诉求走"AI 建草稿 + 人工审批"的工单闭环。

> Beauty e-commerce customer-service AI Agent with Profile+Event dual-channel user memory, a hand-written Function Calling loop, HITL ticket approval, guardrails, and a golden-dataset eval pipeline.

项目源于 Mem0「数据共情者」赛题，重点不在功能多，而在**场景窄但工程闭环完整**：停止条件、失败处理、护栏、轨迹回放、可重复评估，每一层都有明确实现。

## 架构总览

```text
客服/客户对话消息
        │
        ▼
  安全防护（security_guard：PII 脱敏 / 提示注入检测）
        │
        ▼
  Profile+Event 记忆管线（profile_event_pipeline）
    对话 → LLM 抽取 → Schema 校验 → 去重 → SQLite 双通道存储
    - Profile：稳定画像（肤质/过敏史/偏好），固定 24 个 memory_key
    - Event ：带时间戳的行为事件流（purchase / dissatisfaction / experience ...）
        │
        ▼
  上下文组装（context_engine）
    最近对话 + 记忆索引（只给 memory_id/key，不给值 → 强制 Agent 按需取证）
        │
        ▼
  Agent 循环（tool_calling_agent，手写 Function Calling，未用框架）
    while True:
        LLM 决策 → 有 tool_calls？
          ├─ 参数校验 / event_type 枚举校验 / 工单类型语义校验
          ├─ 执行工具（10s 超时，结构化错误回填 role=tool）
          └─ 无 tool_calls → 售后意图护栏检查 → 放行或强制复议（≤2 次）
    工具轮次预算 MAX_ITER=3，耗尽 → 摘除 tools 参数强制收尾（不静默失败）
        │
        ▼
  服务洞察输出（label_engine）
    服务焦点 / 动态标签 / 服务建议，全部引用真实 memory_id 证据
        │
        ▼
  轨迹落盘（agent_trace_store：JSONL，按 run_id 回放任意一次运行的完整消息序列）
```

## 核心设计决策

| 决策 | 理由 |
|---|---|
| 手写 Agent Loop，不用 LangChain 等框架 | 预算控制、护栏注入点、消息协议需要全透明；"LLM 调用次数"与"工具执行轮次"是两个预算，框架里不好拆 |
| 提示词治不好的病用代码治 | 模型曾 3/3 次把"我要退货+泛红"误建成过敏单，改 prompt 无效；改为工具执行前确定性语义校验，一击解决 |
| 先证明尺子准，再量模型 | 评估管道先用确定性规则脚本（零 token）跑 15/15，校准打分逻辑，再跑真实模型 |
| LLM 输出是建议，不是执行 | 唯一写操作（售后工单）只能建 pending 草稿，approve/reject 仅人工可调；数据库 CHECK 约束兜底状态机，形成应用层+存储层纵深防御 |
| 记忆只给索引不给值 | 若全量注入 prompt，模型会跳过工具直接编结论；只给 memory_id/key 列表是"逼"模型走工具取证的工程手段 |

## 评估结果

15 条黄金数据集（`eval/agent_eval_golden.json`），覆盖产品咨询 / 成分安全 / 售后政策 / 退换货 / 过敏投诉 / 历史查询 / 闲聊反例。评估脚本支持确定性基线（零 token 回归）与真实模型多轮重复（`--runs N`）。

qwen-plus 真实评估（3 轮均值）：

| 指标 | 数值 |
|---|---|
| 任务完成率 | **95.6%**（单轮区间 86.7% ~ 100%） |
| 工具选择召回 | 98.9% |
| 参数断言通过率 | 100% |
| 违禁工具误触（如闲聊建单） | 0% |
| 洞察 JSON 可解析率 | 97.8% |
| 平均 LLM 步数 | 2.22 |
| 平均 token / 次 | ~7.5k |
| 漏建单护栏触发率 | 4.4% |

迭代轨迹：13.3% → 86.7% → 60.0% → 84.4% → 95.6%。每一轮失败案例均通过轨迹回放定位根因（JSON 围栏/双份拼接/max_tokens 截断、工单类型误分类、枚举幻觉等），修复后重跑验证。

重要发现：**temperature=0 下同批用例完成率仍有 ±7~13% 波动**——因此评估以多轮均值代替单次数字，这也是 `--runs N` 参数存在的原因。

## 快速开始

环境要求：Windows 10/11、Python 3.11+（开发环境为 3.13）、可访问阿里云百炼（DashScope）的网络。

```powershell
git clone https://github.com/xq-XQ-xqxq/beauty-agent-memory.git
cd beauty-agent-memory
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

配置 API Key（推荐写入用户级环境变量，或项目根目录 `.env`，两者均不入 Git）：

```powershell
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "你的Key", "User")
```

启动客服工作台（Streamlit 三视图：服务洞察 / 用户记忆 / 运行详情）：

```powershell
.\一键启动工作台.bat
# 或
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501
```

访问 http://localhost:8501 。创建会话需填写 `user_id`——它是长期记忆隔离标识，相同 `user_id` 跨会话共享记忆。

运行评估：

```powershell
# 确定性基线：零 token，验证评估脚本与核心链路（应 15/15）
.\.venv\Scripts\python.exe batch_test_agent_eval.py

# 真实模型多轮评估（产生 token 费用，报告自动与上轮对比 delta）
.\.venv\Scripts\python.exe batch_test_agent_eval.py --real --runs 3
```

运行单元测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（或 `MEM0_LLM_API_KEY`） | 必填 |
| `MEM0_LLM_BASE_URL` | OpenAI 兼容端点，可切换任意兼容服务 | DashScope 兼容端点 |
| `MEM0_LLM_MODEL` | 对话模型 | `qwen-plus` |
| `MEM0_EMBED_BASE_URL` | Embedding 端点 | 同上 |

## 项目结构

```text
├─ streamlit_app.py              # 客服工作台（洞察/记忆/运行详情三视图）
├─ Agent 核心
│  ├─ tool_calling_agent.py      # 手写 Function Calling 循环 + 护栏 + 强制收尾
│  ├─ tool_registry.py           # 5 工具注册 / 参数校验 / 超时 / 失败回填
│  ├─ service_ticket_store.py    # 售后工单 HITL 存储（状态机 + CHECK 约束）
│  └─ agent_trace_store.py       # 全链路轨迹 JSONL 落盘与按 run_id 回放
├─ 记忆系统
│  ├─ profile_event_schema.py    # Profile/Event 双通道 schema（固定枚举）
│  ├─ profile_event_pipeline.py  # 对话 → 记忆抽取管线
│  ├─ profile_event_store.py     # SQLite 存储（按 user_id 隔离）
│  ├─ profile_event_dedup.py     # 幂等去重
│  ├─ processed_turn_store.py    # 已处理轮次记录（断点幂等）
│  ├─ memory_lifecycle.py        # 记忆生命周期处理
│  ├─ context_engine.py          # 上下文组装（记忆索引注入）
│  └─ label_engine.py            # Agent 模式服务洞察生成
├─ 检索
│  ├─ knowledge_retriever.py / hybrid_retriever.py / vector_retriever.py
│  └─ knowledge_base/documents.json   # 美妆知识库语料
├─ 安全
│  └─ security_guard.py          # PII 脱敏 / 提示注入检测
├─ 评估
│  ├─ eval/agent_eval_golden.json     # 15 条黄金数据集
│  └─ batch_test_agent_eval.py        # 评估脚本（确定性基线 / --real / --runs N）
└─ tests/                        # 91 项单元测试
```

## 安全与边界

- `.env`、`.venv/`、`data/`（数据库/轨迹/评估报告/向量索引）均已加入 `.gitignore`；
- 对话内容会发送至所配置的 LLM / Embedding 服务，请勿输入真实敏感个人信息；
- 本项目为面试/演示用途的工程原型，不构成医疗建议或生产级客服系统。
