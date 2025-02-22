# tysTech2 智能助手

本项目是一个基于 Langchain 和 NVIDIA AI Endpoints 的智能助手，旨在为用户提供农业、气象和风景旅游方面的服务。

## 功能

- **农业助手**: 回答用户关于农业方面的问题。
- **气象预报**: 根据用户提供的城市名称，生成详细的天气预报解读和生活建议。
- **风景旅游助手**: 回答用户关于风景旅游方面的问题。
- **相似风景推荐**: 根据用户上传的风景图片，推荐相似的风景地点。

## 技术栈

- Langchain
- NVIDIA AI Endpoints
- Gradio

## 文件说明

- `demo.py`: Gradio 界面和主要逻辑。
- `embedding.py`: 负责将文本转换为 embedding 向量，并保存到 FAISS 向量库中。
- `config.py`: 配置文件，包含 API 密钥等信息。

## 使用方法

1. 准备好 NVIDIA API Key 和 Weather API Key，并将其填入 `config.py` 文件中。
2. 运行 `demo.py` 文件，启动 Gradio 界面。
3. 在 Gradio 界面中选择模式，输入问题或城市名称，点击提交按钮，即可获得智能助手的回答。

## 详细说明

### `demo.py`

`demo.py` 文件是整个项目的入口，它包含了 Gradio 界面和主要逻辑。

- 导入必要的库:
    - `langchain_nvidia_ai_endpoints`: 用于连接 NVIDIA AI Endpoints。
    - `langchain.vectorstores`: 用于创建和管理向量数据库。
    - `langchain_core.output_parsers`: 用于解析 LLM 的输出。
    - `langchain_core.prompts`: 用于创建提示模板。
    - `langchain_core.runnables`: 用于构建 RAG 链。
    - `gradio`: 用于创建用户界面。
    - `os`: 用于访问环境变量。
    - `requests`: 用于发送 HTTP 请求。
    - `pandas`: 用于处理数据。
    - `datetime`: 用于处理时间。
    - `base64`: 用于处理图像数据。
    - `langchain_core.messages`: 用于构建消息。
    - `config`: 用于加载 API 密钥。

- 设置 API 密钥:
    - 从环境变量中获取 NVIDIA API Key 和 Weather API Key。
    - 初始化 LLM。

- `print_context(input_data)`: 打印检索到的上下文信息。
    - 输入: `input_data` (包含上下文信息的字典)。
    - 输出: 无。

- `image2b64(image_file)`: 将图像文件转换为 base64 编码。
    - 输入: `image_file` (图像文件路径)。
    - 输出: base64 编码的图像数据。

- `analyze_image_with_ai(image_b64)`: 使用 AI 分析图像。
    - 输入: `image_b64` (base64 编码的图像数据)。
    - 输出: AI 分析结果。

- `initialize_ragbot(embedding_path)`: 初始化 RAG (Retrieval-Augmented Generation) 机器人。
    - 输入: `embedding_path` (embedding 向量库路径)。
    - 输出: RAG 链。
    - 加载 embedding 向量库。
    - 定义提示模板。
    - 构建 RAG 链。

- `get_weather(city='东昌区', is_forecast=True)`: 获取天气数据。
    - 输入: `city` (城市名称), `is_forecast` (是否获取天气预报)。
    - 输出: 天气数据。
    - 从 `AMap_adcode_citycode.xlsx` 文件中获取城市编码。
    - 使用高德地图 API 获取天气数据。

- `format_weather_data(data)`: 格式化天气数据并创建提示。
    - 输入: `data` (天气数据)。
    - 输出: 格式化后的天气数据。

- `generate_weather_forecast(city)`: 生成天气预报。
    - 输入: `city` (城市名称)。
    - 输出: 天气预报。
    - 获取天气数据。
    - 格式化天气数据。
    - 使用 LLM 生成天气预报。

- `agriculture_qa(question)`: 农业问答功能。
    - 输入: `question` (问题)。
    - 输出: 回答。
    - 使用农业 RAG 链回答问题。

- `landscape_qa(question)`: 风景旅游问答功能。
    - 输入: `question` (问题)。
    - 输出: 回答。
    - 使用风景旅游 RAG 链回答问题。

- `landscape_searchqa(image)`: 相似风景推荐功能。
    - 输入: `image` (图像文件路径)。
    - 输出: 相似风景推荐。
    - 使用 AI 分析图像。
    - 使用风景旅游 RAG 链推荐相似风景。

- `main_interface(question, city, landscape_question, landscape_search, mode)`: 主界面函数。
    - 输入: `question` (农业问题), `city` (城市名称), `landscape_question` (风景旅游问题), `landscape_search` (图像文件路径), `mode` (模式)。
    - 输出: 助手回答。
    - 根据模式选择不同的功能。

### `embedding.py`

`embedding.py` 文件负责将文本转换为 embedding 向量，并保存到 FAISS 向量库中。

- 导入必要的库:
    - `langchain.document_loaders`: 用于加载文档。
    - `langchain_nvidia_ai_endpoints`: 用于连接 NVIDIA AI Endpoints。
    - `langchain.text_splitter`: 用于分割文本。
    - `langchain.vectorstores`: 用于创建和管理向量数据库。
    - `os`: 用于访问环境变量。
    - `tqdm`: 用于显示进度条。
    - `json`: 用于处理 JSON 数据。
    - `config`: 用于加载 API 密钥。

- 初始化 embedder:
    - 使用 NVIDIAEmbeddings 初始化 embedder。

- 初始化文本分割器:
    - 使用 CharacterTextSplitter 初始化文本分割器。

- `split_long_texts(splits, max_length=4000)`: 分割长文本。
    - 输入: `splits` (文本片段列表), `max_length` (最大长度)。
    - 输出: 分割后的文本片段列表。
    - 检测 splits 中的每一个长度，如果长度大于 max_length，就强制切分成多个并返回。

- `save_jsonl_as_embedding(jsonl_file, output_folder)`: 将 JSONL 文件转换为 embedding 向量，并保存到 FAISS 向量库中。
    - 输入: `jsonl_file` (JSONL 文件路径), `output_folder` (输出文件夹路径)。
    - 输出: 无。
    - 读取 JSONL 文件。
    - 分割文本。
    - 将文本转换为 embedding 向量。
    - 将 embedding 向量保存到 FAISS 向量库中。

## 依赖

- langchain_nvidia_ai_endpoints
- langchain
- gradio
- pandas
- requests

## 环境变量

- `NVIDIA_API_KEY`: NVIDIA API 密钥
- `WEATHER_API_KEY`: 天气 API 密钥
