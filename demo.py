from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import gradio as gr
import os
import requests
import pandas as pd
import datetime
import base64
from langchain_core.messages import HumanMessage
from config import NVIDIA_API_KEY, WEATHER_API_KEY

# 设置API密钥
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
weatherapikey=WEATHER_API_KEY
llm = ChatNVIDIA(model="qwen/qwen2.5-7b-instruct",max_tokens=4096)
# 加载保存的 embedding


def print_context(input_data):
    context = input_data["context"]
    print("检索到的上下文：")
    for doc in context:
        print(f"来源: {doc.metadata['source']}")
        print(f"内容: {doc.page_content}\n")
    return input_data


def image2b64(image_file):
    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
        return image_b64

def analyze_image_with_ai(image_b64):

        llm2 = ChatNVIDIA(
            model="microsoft/phi-3-vision-128k-instruct",
            temperature=0.7,       # 控制创造性 (0-1)
            max_tokens=2048        # 最大输出长度
        )
        

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "尽可能用中文详细描述你看到的图像的特征。"},
                    {"type": "text", "text": f'<img src="data:image/jpeg;base64,{image_b64}" />'}
                ]
            )
        ]
    
        response = llm2.invoke(messages)
        return response.content



# 初始化RAGbot
def initialize_ragbot(embedding_path):
    # 加载 embedding 向量库
    def load_embedding_store():
        store = FAISS.load_local(embedding_path, NVIDIAEmbeddings(model="baai/bge-m3"), allow_dangerous_deserialization=True)
        return store
    store = load_embedding_store()
    retriever = store.as_retriever(search_kwargs={"k": 10})

    # 初始化 LLM


    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据下文回答问题：\n<Documents>\n{context}\n</Documents>"),
        ("user", "{question}"),
    ])

    # 定义 RAG 链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        # | RunnableLambda(print_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# 初始化


# 获取天气数据的函数
def get_weather(city='东昌区', is_forecast=True):
    try:
        city_code_table = pd.read_excel(r'AMap_adcode_citycode.xlsx')
        matching_rows = city_code_table[city_code_table['中文名'] == city]
        
        if matching_rows.empty:
            return {"error": f"找不到城市 '{city}' 的编码信息"}
            
        adcode = matching_rows.iloc[0]['adcode']
        print(f"找到城市编码: {adcode}")
        
        extensions = 'all' if is_forecast else 'base'
        url = f'https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={weatherapikey}&extensions={extensions}'
        
        response = requests.get(url)
        data = response.json()
        
        if data['status'] != '1':
            return {"error": f"API请求失败: {data.get('info', '未知错误')}"}
            
        return data
    except Exception as e:
        return {"error": f"获取天气数据时出错: {str(e)}"}

# 格式化天气数据并创建提示
def format_weather_data(data):
    if "error" in data:
        return f"错误: {data['error']}"
        
    try:
        if 'forecasts' not in data or not data['forecasts']:
            return "未能获取天气预报数据"
            
        forecast = data['forecasts'][0]
        city = forecast['city']
        province = forecast['province']
        casts = forecast['casts']
        
        weather_info = f"地点: {province}{city}\n"
        weather_info += f"报告时间: {forecast['reporttime']}\n\n"
        
        for cast in casts:
            date = cast['date']
            weekday = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][int(cast['week']) - 1]
            weather_info += f"日期: {date} ({weekday})\n"
            weather_info += f"天气: 白天 {cast['dayweather']}, 夜间 {cast['nightweather']}\n"
            weather_info += f"温度: 白天 {cast['daytemp']}°C, 夜间 {cast['nighttemp']}°C\n"
            weather_info += f"风向: 白天 {cast['daywind']}风, 夜间 {cast['nightwind']}风\n"
            weather_info += f"风力: 白天 {cast['daypower']}级, 夜间 {cast['nightpower']}级\n\n"
        
        return weather_info
    except Exception as e:
        return f"格式化天气数据时出错: {str(e)}"

# 生成天气预报
def generate_weather_forecast(city):
    if not city:
        return "请输入城市名称"
    
    # 获取天气数据
    weather_data = get_weather(city, True)
    formatted_weather = format_weather_data(weather_data)
    print(formatted_weather)
    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的气象预报员。请根据以下天气数据为用户提供详细、准确的天气预报解读，并给出合理的生活建议。"),
        ("user", f"请根据以下天气数据，生成一份详细的天气预报解读和生活建议：\n\n{formatted_weather}")
    ])
    
    # 使用LLM生成天气预报
    forecast_chain = prompt | llm | StrOutputParser()
    forecast = forecast_chain.invoke({})
    
    return forecast


agriculture_rag_chain = initialize_ragbot(embedding_path='nongye/')
landscape_rag_chain = initialize_ragbot(embedding_path='fengjing/')



# 农业问答功能
def agriculture_qa(question):
    if not question:
        return "请输入您的农业问题"
    response = agriculture_rag_chain.invoke(question)
    return response

def landscape_qa(question):
    if not question:
        return "请输入您的风景旅游问题"
    response = landscape_rag_chain.invoke(question)
    return response

def landscape_searchqa(image):
    # if not image:
    #     return "请输入您的风景图片"
    image_b64 = image2b64(image)
    result=analyze_image_with_ai(image_b64=image_b64)
    response = landscape_rag_chain.invoke("你是一个风景匹配助手，任务是根据用户的描述，匹配几个适合的风景，并说出匹配理由，回答尽可能详细，匹配到的风景必须是具体的地点：\n回答格式:\n相似的风景如下\n风景1名称:风景1描述\n风景2名称:风景2描述\n风景3名称:风景3描述......\n\n以下是用户的描述\n\n"+result)
    return response

def main_interface(question, city, landscape_question, landscape_search, mode):
    if mode == "农业助手":
        return agriculture_qa(question), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif mode == "气象预报":  # 气象预报
        return generate_weather_forecast(city), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif mode == "风景旅游助手":
        return landscape_qa(landscape_question), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif mode == "相似风景推荐":
        return landscape_searchqa(landscape_search), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

# 创建 Gradio 界面
with gr.Blocks(title="tysTech2", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# tysTech2 智能助手")
    
    with gr.Row():
        mode = gr.Radio(["农业助手", "气象预报", "风景旅游助手", "相似风景推荐"], label="选择模式", value="农业助手")
    
    with gr.Row():
        # 农业助手输入
        agriculture_input = gr.Textbox(label="请输入您的农业问题", visible=True)
        
        # 气象预报输入
        weather_input = gr.Textbox(label="请输入城市名称 (例如: 东昌区)", visible=False)
        landscape_input = gr.Textbox(label="请输入您的风景旅游问题", visible=False)
        landscape_search = gr.Image(label="请放入您想寻找的风景图片", visible=False, type="filepath")
    
    with gr.Row():
        submit_button = gr.Button("提交")
    
    output = gr.Textbox(label="助手回答")
    
    # 当模式改变时更新界面
    mode.change(
        fn=lambda m: (
            gr.update(visible=(m == "农业助手")),
            gr.update(visible=(m == "气象预报")),
            gr.update(visible=(m == "风景旅游助手")),
            gr.update(visible=(m == "相似风景推荐"))
        ),
        inputs=[mode],
        outputs=[agriculture_input, weather_input, landscape_input, landscape_search]
    )
    
    # 提交按钮点击事件
    submit_button.click(
        fn=main_interface,
        inputs=[agriculture_input, weather_input, landscape_input, landscape_search, mode],
        outputs=[output, agriculture_input, weather_input, landscape_input, landscape_search]
    )

# 主界面函数


# 启动 Gradio 界面
if __name__ == "__main__":
    # print(ChatNVIDIA.get_available_models())
    interface.launch(debug=True, share=False, server_port=5124, server_name="127.0.0.1")