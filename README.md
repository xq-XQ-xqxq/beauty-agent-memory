# 共情记忆客服工作台

面向美妆电商人工客服的 AI 辅助原型。系统读取当前对话并维护用户长期记忆，再将当前消息、最近对话和完整用户记忆映射为客服此刻最应该看到的服务焦点、动态标签和有限服务建议。

## 核心模块

### 1. 用户事实记忆引擎

```text
客户对话
  → 事实提取 Agent
  → Schema、证据和敏感信息校验
  → 记忆决策 Agent（ADD / UPDATE / DELETE / NOOP）
  → Mem0 + Qdrant 长期记忆
```

- 使用固定的 24 个 `memory_key`，LLM不能自行新增槽位；
- 原始消息作为证据保留；
- 记忆按 `user_id` 隔离，不按会话或事件隔离；
- 第二次校验数据库返回记录的 `user_id`，避免跨用户数据进入模型。

### 2. 客服服务洞察引擎

```text
当前消息 + 最近对话 + 当前用户全部有效记忆
  → 单次 LLM 调用
  → 当前服务焦点 + 最多5个动态标签 + 最多2条服务建议
```

- 不使用查询Tool，不进行二次向量检索；
- 标签标题和内容由模型动态决定；
- 每个标签与建议必须引用真实的消息ID或记忆ID；
- 服务建议仅限沟通辅助，不做医学诊断、业务审批或赔偿承诺。

## 工作台

Streamlit 工作台采用三栏布局：

- 左侧：会话列表、新建会话与测试工具；
- 中间：人工聊天区，可切换由“客户”或“客服”发送消息；
- 右侧：当前服务焦点、动态标签、AI建议和记忆依据。

创建会话时需要填写 `user_id`。相同 `user_id` 的不同会话共享长期记忆，不同 `user_id` 的记忆相互隔离。

## 运行环境

建议准备：

- Windows 10/11；
- Python 3.11 或 3.12（64位）；
- 可访问阿里云 DashScope 的网络；
- 一个有效的 `DASHSCOPE_API_KEY`；
- 约 1GB 可用磁盘空间。

主要依赖：

- `mem0ai==2.0.15`
- `qdrant-client==1.18.0`
- `streamlit==1.60.0`
- `python-dotenv==1.2.2`

模型默认使用：

- LLM：`qwen-plus`
- Embedding：`text-embedding-v4`，1024维

## 安装

```powershell
git clone https://gitee.com/ffqws/oulaiya.git
cd oulaiya

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，可以不激活虚拟环境，直接执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置 DashScope Key

推荐保存为当前Windows用户的环境变量，不需要管理员权限：

```powershell
[Environment]::SetEnvironmentVariable(
    "DASHSCOPE_API_KEY",
    "替换成你的Key",
    "User"
)
```

设置后请重新打开 PowerShell 或重新双击启动脚本。

也可以只对当前终端临时设置：

```powershell
$env:DASHSCOPE_API_KEY="替换成你的Key"
```

不要把真实Key写入代码或提交到Git。

## 启动

最简单的方式是双击：

```text
一键启动工作台.bat
```

也可以在 PowerShell 中运行：

```powershell
.\run_demo.ps1
```

或直接启动 Streamlit：

```powershell
$env:DASHSCOPE_API_KEY="替换成你的Key"
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501
```

浏览器访问：

```text
http://localhost:8501
```

首次运行时会自动创建本地 `data/` 目录，用于保存 Qdrant、Mem0运行数据、幂等记录和日志。该目录不会提交到Git。

## 测试

安装测试依赖：

```powershell
pip install -r requirements-dev.txt
```

运行测试：

```powershell
pytest -q -p no:cacheprovider
```

当前测试覆盖固定记忆Schema、证据门控、ADD/UPDATE/DELETE/NOOP、用户隔离、第二次映射输出校验和前端多会话隔离。

## 重要测试说明

- `user_id` 是长期记忆隔离标识，不是普通昵称；
- 使用曾经测试过的 `user_id` 会读取该用户以前保存的记忆；
- 测试全新用户时，请使用从未使用过的ID，例如 `test_buyer_A_001`；
- 相同用户跨会话共享记忆是预期行为；
- 当前服务焦点和标签需要手动点击右侧“分析”生成。

## 数据与安全

- `.env`、`.venv/`、`data/`、缓存和日志均已加入 `.gitignore`；
- 默认关闭 Mem0 遥测；
- Demo对话会发送给DashScope模型和Embedding服务，请勿输入真实敏感信息；
- 本项目用于黑客松原型演示，不应直接作为生产客服或医疗建议系统使用。

## 主要文件

- `streamlit_app.py`：三栏客服工作台；
- `memory_schema.py`：24个固定记忆槽位；
- `evidence_pipeline.py`：事实提取与证据校验；
- `memory_decision.py`：记忆变更决策；
- `memory_service.py`：Mem0/Qdrant封装与用户隔离；
- `label_engine.py`：第二次服务洞察映射；
- `processed_turn_store.py`：幂等处理记录；
- `tests/`：自动化测试。
