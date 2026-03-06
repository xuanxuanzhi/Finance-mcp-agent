# 股票投资顾问 Agent｜个性化学习笔记（进行中）

> 目标：不仅跑通项目，更要理解“为什么”。本笔记会持续补全：架构、关键术语、模块作用、常见报错与排查、以及面试高频问答。

## 1. 我对项目的整体理解（先写人话版本）

- **这套系统在做什么**：输入一句自然语言（如“帮我看看茅台 600519 值得投资吗”），系统会并行做四类分析（基本面/技术面/估值/新闻），最后汇总成一份结构化 Markdown 报告。
- **为什么要多 Agent**：把复杂任务拆成“可解释的小模块”，每个模块用工具拿到真实数据，再由大模型做推理与总结，最后统一汇总，结果更稳定也更可控。

## 2. 项目架构速览（我能画出来）

- **MCP Server（工具提供方）**：`a-share-mcp-is-just-i-need/`
  - 作用：把 A 股数据能力（行情、财报、宏观、新闻爬虫等）包装成“工具”，给上层 Agent 调用。
- **MCP Client + LangGraph 工作流（编排方）**：`Financial-MCP-Agent/`
  - 作用：定义 5 个 Agent 节点，并行执行 4 个分析 Agent，最后由总结 Agent 生成报告。

## 3. 运行与测试 checklist（每步都可验证）

- [ ] 3.1 创建并激活 Python 环境（建议 conda + Python 3.12）
- [ ] 3.2 安装依赖（`pip install -r requirements.txt`）
- [ ] 3.3 配置 `.env`（模型 API Key / Base URL / Model）
- [ ] 3.4 修正 MCP 服务器路径（`Financial-MCP-Agent/src/tools/mcp_config.py`）
- [ ] 3.5 先测 MCP Server 数据源（`python test_baostock.py`）
- [ ] 3.6 再跑 Agent 主程序（`python src/main.py --command "..."`）

### 3.7 我这台机器的“跑通证据”（要能复述给面试官/同学听）

- **LLM 连通性极简测试**：用 1 句话让模型只回复 `OK`，验证 Key/URL/Model 都没问题（避免跑整套工作流才发现余额/权限错误）。
- **端到端报告生成成功**：报告路径示例：
  - `Financial-MCP-Agent/reports/report_茅台_600519_20260306_155208.md`

## 4. 关键术语（遇到一个写一个）

- **LLM（大语言模型）**：像“会写分析报告的大脑”，但它不自带行情/财报数据，必须通过工具拿到真实数据。
- **Tool（工具）**：像“按钮/插件”。例如“获取K线”“拉取财报”“爬新闻”。LLM 通过工具把“猜测”变成“基于数据”。
- **MCP（Model Context Protocol）**：一种“让大模型调用外部工具”的标准协议。可以把它理解为：LLM 和工具之间的“USB 口”。
- **Agent**：会自己决定下一步要做什么、要不要调用工具的“助理”。
- **LangGraph**：像“流程编排器”。规定多个 Agent 怎么并行、怎么汇聚、怎么结束。
- **ReAct**：一种常用的 Agent 工作方式：先“想”（Reasoning），再“做”（Act，调用工具），反复直到得到答案。

### 4.1 ReAct 与 Agent 搭建（回答“是封装好的吗、还有别的模式吗”）

- **流程**：先创建 LLM 实例，再拿到 MCP 工具列表，然后 `create_react_agent(llm, mcp_tools)` 就把“大模型 + 工具”绑在一起，形成“思考 → 决定是否调工具 → 拿到结果再思考”的循环。是的，**ReAct 模式在本项目里是封装好的**（LangGraph 的 `create_react_agent`），不需要自己写循环。
- **是否只有 ReAct**：不是。LangGraph 里可以自己用 `StateGraph` 手写“先调工具再调 LLM”的节点和边，或者用别的库（如 LangChain 的 AgentExecutor、AutoGPT 式规划等）。ReAct 只是最常用、最易上手的一种“推理+行动”模式。
- **面试可答**：我们用的是 LangGraph 的 ReAct 预构建 Agent，把 LLM 和 MCP 工具绑在一起，由模型自主决定何时调哪个工具、调几次，直到得出分析结论。

### 4.2 MCP 在本项目里的角色（谁才是 Server、谁在注册、谁在实现）

- **一句话**：`a-share-mcp-is-just-i-need` 是**我们自己建的 MCP 服务器**，不是用网上现成的 MCP 服务；工具也是我们自己在服务器里注册和实现的。
- **具体分层**：
  - **MCP 服务器（我们自己的）**：项目 `a-share-mcp-is-just-i-need`，入口 `mcp_server.py`，用 FastMCP 起一个进程，用 stdio 和客户端通信。
  - **在服务器里“注册”工具**：`src/tools/news_crawler.py` 里用 `@app.tool()` 把 `crawl_news` 注册到这台 MCP 服务器上，所以这里是**在咱们自己的 MCP 服务器上注册工具**，不是用网上现成的 MCP 工具。
  - **工具的具体实现**：`baostock_data_source.py` 里的 `crawl_news()` 是**工具的真实逻辑**（爬百度新闻、解析、调情感/风险模型等），跑在 MCP 服务器进程里，是**服务端、本地实现**。`news_crawler.py` 里的 `crawl_news` 只是薄薄一层，内部调 `data_source.crawl_news(query, top_k)`。
  - **MCP 客户端**：`Financial-MCP-Agent` 里的 `mcp_client.py` 通过 `MultiServerMCPClient` 连到上面的 MCP 服务器，拿到“工具列表”，再交给 LangGraph 的 ReAct Agent 去调用。所以 Agent 侧是**调用方（客户端）**，不关心工具内部是爬虫还是查库。
- **面试怎么说**：我们自己做了一个 A 股 MCP 服务器，把行情、财报、新闻爬虫等封装成工具并在服务器里注册；Agent 端通过 MCP 协议连上这台服务器，拿到工具列表再按需调用。**工具内部怎么实现**（例如 crawl_news 里怎么爬百度、怎么调情感模型）可以一句带过；**面试更关注**你为什么用 MCP、谁当 Server 谁当 Client、怎么用 MCP 把“大模型”和“数据能力”接在一起，而不是逐行讲爬虫代码。

## 5. 面试高频问题（问题 → 对应代码 → 我的回答）

> 我会把你 PDF 里的问题逐条映射到文件与函数上，保证你能“指着代码解释”。

- Q1：新闻分析 Agent 的数据来源是什么？
  - 对应代码：`a-share-mcp-is-just-i-need/src/tools/news_crawler.py`（新闻抓取工具注册）、`a-share-mcp-is-just-i-need/src/baostock_data_source.py` 的 `crawl_news()`（具体爬取实现）
  - **我的概括（人话）**：新闻里显示资金大量流出，情感偏负，建议短期观望，同时要结合公司基本面综合判断。
  - **面试版标准回答**：新闻分析的数据来自我们封装的 MCP 工具 `crawl_news`，线上实时来源是百度新闻搜索，抓取与公司相关的标题、摘要和链接，部分会抓正文；情感和风险分数由我们本地微调的小模型（或规则）给出。报告里会根据资金流向、情感和风险给出短期观望或结合基本面再决策的建议。
  - **口径区分**：若问“训练情感/风险模型的数据从哪来”，答东方财富/新浪/证券时报/第一财经等采集并标注的那套离线数据集，与线上爬虫是两套口径。

## 6. 我遇到的坑与解决（按时间记录）

- （待记录）

### 6.1 常见坑：`No module named 'src'`

- **现象**：直接运行 `python src/main.py ...` 报 `ModuleNotFoundError: No module named 'src'`
- **原因（人话）**：Python 不知道去哪里找 `src/` 这个包（就像你在电脑里不告诉它“这个文件夹也是搜索路径”）。
- **解决**：在 `src/main.py` 最开头把项目根目录加入 `sys.path`，使两种运行方式都可用：
  - `python src/main.py ...`
  - `python -m src.main ...`

