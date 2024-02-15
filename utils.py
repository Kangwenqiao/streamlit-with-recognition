#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def _display_detected_frames(conf, model, st_frame, image):
    """
    使用YOLOv8模型在视频帧上显示检测到的对象。
    :param conf (float): 对象检测的置信度阈值。
    :param model (YOLOv8): 包含YOLOv8模型的`YOLOv8`类的实例。
    :param st_frame (Streamlit对象): 用于显示检测视频的Streamlit对象。
    :param image (numpy数组): 表示视频帧的numpy数组。
    :return: None
    """
    # 将图像调整为标准大小
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用YOLOv8模型预测图像中的对象
    res = model.predict(image, conf=conf)

    # 在视频帧上绘制检测到的对象
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='检测视频',
                   channels="BGR",
                   use_column_width=True
                   )

@st.cache_resource
def load_model(model_path):
    """
    从指定的model_path加载YOLO对象检测模型。

    参数:
        model_path (str): YOLO模型文件的路径。

    返回:
        一个YOLO对象检测模型。
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    对上传的图片执行推理
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="选择一张图片...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="上传的图片",
                use_column_width=True
            )

    if source_img:
        if st.button("开始执行"):
            with st.spinner("执行中..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="检测结果",
                             use_column_width=True)
                    try:
                        with st.expander("检测结果详情"):
                            for box in boxes:
                                st.write(f"位置及大小: {box.xywh}")
                    except Exception as ex:
                        st.write("尚未上传图片！")
                        st.write(ex)

def infer_uploaded_video(conf, model):
    """
    对上传的视频执行推理
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="选择一个视频..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("开始执行"):
            with st.spinner("执行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"加载视频出错: {e}")

def infer_uploaded_webcam(conf, model):
    """
    执行网络摄像头的推理。
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """
    try:
        flag = st.button("停止运行")
        vid_cap = cv2.VideoCapture(0)  # 本地摄像头
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"加载视频出错: {str(e)}")
