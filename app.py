#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
from pathlib import Path

import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam


# 将图片文件转换为Base64编码的函数
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 获取背景图片的Base64编码
background_image_base64 = get_image_base64("back_images/image2.jpg")

# 设置页面配置
st.set_page_config(
    page_title="飞机检测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 使用markdown添加自定义CSS样式来设置背景图像，并添加遮光罩
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{background_image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* 添加遮光罩 */
.stApp:before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5); /* 黑色遮光罩，透明度为50% */
    z-index: -1;
}}

/* 调整其他元素以确保遮光罩效果 */
h1, h2, h3, .stButton>button, .stTextInput>div>div>input {{
    position: relative;
    z-index: 1;
}}

/* 其他样式 */

h1{{
    color: #FFF;
    font-family: '楷体';
}}
.stSidebar {{
    background-color: #FDEDEC;
}}
.stButton>button {{
    color: #FFFFFF;
    background-color: #FF4500;
    border-radius: 20px;
    border: 1px solid #FF4500;
}}
.stTextInput>div>div>input {{
    color: #8B0000;
}}
</style>
""", unsafe_allow_html=True)

# 主页面标题
st.title("飞机检测")

# 侧边栏配置和其余代码
# 侧边栏
st.sidebar.header("深度学习模型配置")

# 选择任务类型
task_type = st.sidebar.selectbox(
    "选择任务类型",
    ["检测"]
)

model_type = None
if task_type == "检测":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("当前仅实现了'检测'功能")

confidence = float(st.sidebar.slider(
    "选择模型置信度", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("请在侧边栏中选择模型")

# 加载预训练的深度学习模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"无法加载模型，请检查指定的路径：{model_path}")

# 图像/视频选项
st.sidebar.header("图像/视频配置")
source_selectbox = st.sidebar.selectbox(
    "选择数据源",
    config.SOURCES_LIST
)

if source_selectbox == config.SOURCES_LIST[0]:  # 图像
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]:  # 视频
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]:  # 摄像头
    infer_uploaded_webcam(confidence, model)
else:
    st.error("当前仅实现了'图像'和'视频'数据源")
