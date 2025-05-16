# 使用 Elasticsearch 和 Milvus 构建 RAG 系统

本项目包含两个独立的 Jupyter Notebook，分别演示了如何使用 Elasticsearch 和 Milvus 构建检索增强生成 (RAG) 系统。RAG 系统通过结合大规模语言模型的生成能力和外部知识库的检索能力，来提供更准确、更相关的回答。

## 项目概述

### 1. Elasticsearch 构建 RAG 系统 (`rag_elasticsearch.ipynb`)

这个 Notebook 展示了如何利用 Elasticsearch 作为向量数据库来构建一个电影推荐的 RAG 系统。主要步骤包括：

* **环境准备**: 安装必要的 Python 库，如 `elasticsearch`, `sentence_transformers`, `transformers`, `eland` 和 `datasets`。
* **数据加载与预处理**:
  * 从 Hugging Face Hub 下载 `MongoDB/embedded_movies` 数据集。
  * 对数据进行清洗，移除缺少关键信息（如 `fullplot`）的数据点，并移除已有的 `plot_embedding`（如果存在）。
* **Embedding 模型**:
  * 选择并加载一个 Sentence Transformer 模型 (例如 `thenlper/gte-small`) 用于将电影的剧情文本转换为向量嵌入。
  * 定义函数将文本转换为 embedding。
* **创建 Elasticsearch 索引**:
  * 连接到 Elasticsearch 实例（需要配置 Elasticsearch 主机和 API 密钥）。
  * 定义索引的 mapping，包括文本字段和用于存储向量的 `dense_vector` 字段。
  * 如果索引已存在，则删除旧索引并创建新索引。
* **数据上传到 Elasticsearch**:
  * 将包含文本和对应 embedding 的数据集批量上传到 Elasticsearch 索引中。
* **向量搜索与结果整合**:
  * 定义函数执行向量搜索，根据用户查询的 embedding 找到相似的电影剧情。
  * 将搜索结果与用户原始查询结合，形成更丰富的上下文，用于后续的生成任务（虽然 Notebook 中未直接展示生成部分，但已准备好输入）。
  * 示例查询展示了如何根据剧情描述推荐相关的浪漫电影。

### 2. Milvus 构建 RAG 系统 (`rag_milvus.ipynb`)

这个 Notebook 展示了如何利用 Milvus 作为向量数据库来构建一个基于 PDF 文档问答的 RAG 系统。主要步骤包括：

* **环境准备**: 安装必要的 Python 库，如 `pymilvus`, `sentence-transformers`, `huggingface-hub`, `langchain_community`, `langchain-text-splitters`, `pypdf` 和 `tqdm`。
* **数据加载与处理**:
  * 下载一个 PDF 文件 (例如 "The-AI-Act.pdf")。
  * 使用 `PyPDFLoader` 从 PDF 中提取文本内容。
  * 使用 `RecursiveCharacterTextSplitter` 将提取的文本分割成更小的、带有重叠的文本块 (chunks)。
* **Embedding 模型**:
  * 选择并加载一个 Sentence Transformer 模型 (例如 `BAAI/bge-small-en-v1.5`) 用于将文本块转换为向量嵌入。
  * 定义函数将文本转换为 embedding。
* **数据加载到 Milvus**:
  * 连接到 Milvus 实例 (可以使用 Milvus Lite 将数据存储在本地文件，或连接到远程 Milvus 服务器/Zilliz Cloud)。
  * 定义一个 collection 名称。
  * 如果 collection 已存在，则删除旧 collection。
  * 创建一个新的 collection，指定维度 (与 embedding 模型输出维度一致)、度量类型 (如内积 `IP`) 和一致性级别。
  * 将文本块及其对应的 embedding 批量插入到 Milvus collection 中。
* **构建 RAG - 检索**:
  * 定义一个用户查询 (例如 "What is the legal basis for the proposal?")。
  * 将用户查询转换为 embedding。
  * 在 Milvus collection 中执行向量搜索，检索与查询 embedding 最相似的文本块。
  * 输出检索到的文本块及其与查询的相似度得分。

## 如何运行

1. **克隆/下载项目**: 将包含这两个 Notebook 的文件夹下载到本地。
2. **安装依赖**:
   * 对于 `rag_elasticsearch.ipynb`，请确保安装了 Notebook 中第一部分列出的所有库。您可能需要一个正在运行的 Elasticsearch 实例，并相应地配置 `ELASTICSEARCH_HOST` 和 `ELASTICSEARCH_API_KEY`。
   * 对于 `rag_milvus.ipynb`，请确保安装了 Notebook 中第一部分列出的所有库。Milvus Lite 会在本地创建数据库文件，无需额外配置。如果使用远程 Milvus，请相应修改连接参数。
3. **运行 Jupyter Notebook**:
   * 打开 Jupyter Notebook 或 JupyterLab。
   * 导航到项目文件夹。
   * 分别打开并按顺序运行 `rag_elasticsearch.ipynb` 和 `rag_milvus.ipynb` 中的单元格。

## 注意事项

* **Elasticsearch 配置**: `rag_elasticsearch.ipynb` 需要您提供 Elasticsearch 的主机地址和 API 密钥。请在相应的单元格中替换占位符。
* **Milvus URI**: `rag_milvus.ipynb` 默认使用 Milvus Lite 并将数据存储在本地 `./hf_milvus_demo.db` 文件中。您可以根据需要修改 `MilvusClient` 的 `uri` 参数以连接到不同的 Milvus 实例。
* **Embedding 模型下载**: 首次运行代码时，Sentence Transformer 模型会被下载到本地缓存。
* **计算资源**: 生成 embedding 可能需要一定的计算时间，尤其是在没有 GPU 的情况下。`rag_milvus.ipynb` 中提到了可以使用 GPU 加速 (`device="cuda"`)，如果您的环境支持。

希望这个 README 对您有所帮助！
