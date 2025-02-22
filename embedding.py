from langchain.document_loaders import PyPDFLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from tqdm import tqdm
import json
from config import NVIDIA_API_KEY
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
# 初始化 embedder
embedder = NVIDIAEmbeddings(model="baai/bge-m3")

# 初始化文本分割器
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")




# # 获取 wenxian 文件夹中的所有 PDF 文件
# pdf_files = [f for f in os.listdir("wenxian") if f.endswith('.pdf')]

# # 存储所有文档的文本和元数据
# docs = []
# metadatas = []

# # 遍历所有 PDF 文件
# for pdf_file in pdf_files:
#     # 加载 PDF 文件
#     loader = PyPDFLoader(f"wenxian/{pdf_file}")
#     documents = loader.load()
    
#     # 对每个文档进行分割
#     for doc in documents:
#         splits = text_splitter.split_text(doc.page_content)
#         docs.extend(splits)
#         metadatas.extend([{"source": pdf_file}] * len(splits))

# # 将文本转换为 embedding 并保存到 FAISS 向量库中
# store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
# store.save_local('embedding2/')

# print("所有 PDF 文件的文本已成功提取并转换为 embedding。")

def split_long_texts(splits, max_length=4000):
    """
    检测splits中的每一个长度，如果长度大于max_length，就强制切分成多个并返回。
    
    :param splits: List[str] - 文本片段列表
    :param max_length: int - 每个片段的最大长度
    :return: List[str] - 处理后的文本片段列表
    """
    new_splits = []
    for text in splits:
        if len(text) > max_length:
            # 计算需要切分的次数
            num_splits = (len(text) + max_length - 1) // max_length
            for i in range(num_splits):
                start = i * max_length
                end = start + max_length
                new_splits.append(text[start:end])
        else:
            new_splits.append(text)
    return new_splits

def save_jsonl_as_embedding(jsonl_file, output_folder):
    # 存储所有文档的文本和元数据
    docs = []
    metadatas = []

    # 读取 jsonl 文件
    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            json_obj = json.loads(line)
            text = str(json_obj)  # 假设文本字段名为 'text'
            source = json_obj.get('title', '')  # 假设源字段名为 'source'
            
            # 对每个文档进行分割
            splits = text_splitter.split_text(text)
            splits=split_long_texts(splits)
            docs.extend(splits)
            metadatas.extend([{"source": source}] * len(splits))
    # lens1=[]
    # for i in docs:
    #     lens1.append(len(i))
    # print(max(lens1))
    # 将文本转换为 embedding 并保存到 FAISS 向量库中
    store = FAISS.from_texts(docs[:50], embedder, metadatas=metadatas[:50])
    for i in tqdm(range(50,len(docs),50)):
        store.add_texts(docs[i:i+50], metadatas=metadatas[i:i+50])
        store.save_local(output_folder)

    print(f"文件 {jsonl_file} 的文本已成功提取并转换为 embedding，保存到 {output_folder}。")
if __name__ == "__main__":
    # 使用示例
    jsonl_file = 'data/农业.jsonl'  # 替换为你的 jsonl 文件路径
    output_folder = 'nongye/'  # 替换为你想要保存 embedding 的文件夹路径
    save_jsonl_as_embedding(jsonl_file, output_folder)