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

## 5. 面试高频问题（问题 → 对应代码 → 我的回答）

> 我会把你 PDF 里的问题逐条映射到文件与函数上，保证你能“指着代码解释”。

- Q1：新闻分析 Agent 的数据来源是什么？
  - 对应代码：`a-share-mcp-is-just-i-need/src/tools/news_crawler.py`（新闻抓取工具注册与实现）
  - 进一步定位：`a-share-mcp-is-just-i-need/src/baostock_data_source.py` 的 `crawl_news()`（具体爬取实现）
  - 我的回答（待补全）：
    - **线上运行时（本项目实际爬取）**：通过 `crawl_news` 工具走 `BaostockDataSource.crawl_news()`，用“百度新闻搜索”抓取标题/摘要/链接，再尝试抓正文内容（抓不到就退化为摘要）。
    - **离线训练时（简历/面试材料里那套 10 万新闻数据）**：常见说法是从东方财富/新浪财经/证券时报/第一财经等站点采集并做去重与标注，用来训练新闻情感/风险小模型；这套“训练集来源”与“线上爬虫实时来源”不是一回事，面试时要先澄清你在回答哪个口径。

## 6. 我遇到的坑与解决（按时间记录）

- （待记录）

### 6.1 常见坑：`No module named 'src'`

- **现象**：直接运行 `python src/main.py ...` 报 `ModuleNotFoundError: No module named 'src'`
- **原因（人话）**：Python 不知道去哪里找 `src/` 这个包（就像你在电脑里不告诉它“这个文件夹也是搜索路径”）。
- **解决**：在 `src/main.py` 最开头把项目根目录加入 `sys.path`，使两种运行方式都可用：
  - `python src/main.py ...`
  - `python -m src.main ...`

