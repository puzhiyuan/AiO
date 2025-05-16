# AiO

👋 欢迎来到 AiO！本仓库致力于收集和分享关于大型语言模型（LLM）实际应用的教程、代码示例和最佳实践。

## ✨ 项目简介

随着大型语言模型的飞速发展，它们在各个领域的应用潜力日益显现。本项目旨在为开发者、研究人员和爱好者提供一个学习和实践 LLM 应用的平台，内容涵盖从基础概念到高级技巧的各种教程。

我们希望通过这些教程，帮助大家更好地理解和运用 LLM 技术，解决实际问题，创造更多价值。

## 🎯 目标受众

* 对大型语言模型应用感兴趣的开发者。
* 希望将 LLM 技术应用到自己工作或项目中的研究人员和工程师。
* 正在学习人工智能和自然语言处理的学生。
* 任何对探索 LLM 前沿应用充满热情的人。

## 📚 教程内容

本仓库计划涵盖（并逐步完善）以下主要方向的教程：

* **检索增强生成 (Retrieval Augmented Generation - RAG):**
  * 利用外部知识库提升 LLM 回答的准确性和相关性。
  * 示例：使用 Elasticsearch 构建 RAG (已有 `rag_elasticsearch.ipynb`)
  * 示例：使用 Milvus 构建 RAG (已有 `rag_milvus.ipynb`)
  * 不同向量数据库的选择和应用。
  * 文本切分和 Embedding 策略。
* **AI 智能体 (Agents):**
  * 构建能够自主规划、执行任务的 LLM Agent。
  * LangChain, AutoGen 等框架的使用。
  * Tool Calling 和 Function Calling。
  * 多智能体协作。
* **模型微调 (Fine-tuning):**
  * 在特定任务或数据集上微调预训练的 LLM。
  * LoRA, QLoRA 等高效微调技术。
  * 数据集准备和格式化。
  * 微调流程和超参数调优。
* **模型部署 (Model Deployment):**
  * 将训练好的 LLM 或微调后的模型部署到生产环境。
  * 常见的部署框架和工具 (如 vLLM, TGI, BentoML, FastAPI)。
  * 模型量化和优化，以提高推理效率。
  * API 设计和服务封装。
* **提示工程 (Prompt Engineering):**
  * 设计高效的提示词以引导 LLM 输出期望的结果。
  * 各种高级提示技巧 (如 Few-shot, Chain-of-Thought, Self-Consistency)。
* **模型评估 (Model Evaluation):**
  * 评估 LLM 在不同任务上的表现。
  * 常用的评估指标和工具。
* **(未来可能增加更多主题，例如：LLM 安全与伦理、多模态 LLM 应用等)**

## 📁 仓库结构

为了方便查找和学习，本仓库将主要按以下结构组织：
