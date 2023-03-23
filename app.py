import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
from pyrsistent import s
import wget
import time
import pathlib
from fastai.vision.all import (
    load_learner,
    PILImage,
)
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

def imageInput(device, src):
    enter=False
    description = ["""โรคใบจุดตากบ (frog_eye) จะพบประจำในพริกและยาสูบ จุดแผลจะกลม กลางแผลมีสีเทาขอบแผลสีน้ำตาล แผลจะกระจายทั่วไป โรคนี้เกิดจากเชื้อเซอโคสะปอร่า (Cercospora) การควบคุมโรคนี้ใช้สารชนิดเดียวกับที่ใช้ควบคุมแอนแทรคโนสก็ได้ หรือจะใช้สารประเภทคลอโรธาโรนิล (chlorothalonil) ฉีดพ่นสม่ำเสมอขณะระบาด จะได้ผลดี""", 
                   """โรคใบเหลือง (yellow_leaf) สาเหตุส่วนใหญ่ที่ทำให้ต้นไม้ใบเหลือง เป็นเพราะรดน้ำมากเกินไป ความชื้นในดินสูง หรือดินแน่น ระบายน้ำยาก วิธีแก้ปัญหาเบื้องต้นคือ ตัดใบส่วนนั้นทิ้งไป จากนั้นเว้นการรดน้ำไปสักระยะ แล้วค่อยกลับมารดน้ำใหม่เมื่อดินแห้ง เช็กง่าย ๆ โดยใช้นิ้วกดลงไปในดินประมาณ 1 นิ้ว หากหน้าดินแห้งก็รดน้ำได้ แต่ถ้าดินยังแฉะก็ควรรอก่อน"""]
    GOOD_OR_BAD = pathlib.Path("models/good_or_bad.pkl")
    good_or_bad = load_learner(GOOD_OR_BAD)
    if src == 'จากเครื่องของคุณ':
        image_file = st.file_uploader("อัปโหลดภาพ", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            # call Model prediction--
            checkleaf = good_or_bad.predict(imgpath)
            if checkleaf[0] == 'good' :
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
                # model.cuda() if device == 'cuda' else model.cpu()
                model.cpu()
                pred = model(imgpath)
                class_names = pred.names
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)
                # --Display predicton
                img_ = Image.open(outputpath)
                with col2:
                    st.image(img_, caption='Model Prediction', use_column_width='always')
                st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบจุดตากบ</h1></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[0]}</div>", unsafe_allow_html=True)
                st.markdown("<div><br></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบเหลือง</h1></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[1]}</div>", unsafe_allow_html=True)
            else:
                st.error('ไม่สามารถใช้ภาพนี้ได้ กรุณาอัปโหลดใหม่อีกครั้ง')
    elif src == 'จากไฟล์ของเว็บไซต์':
        # Image selector slider
        test_images = os.listdir('data/images/')
        test_image = st.selectbox('เลือกภาพของเว็บไซต์:', test_images)
        image_file = 'data/images/' + test_image
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
                pred = model(image_file)
                #disease_name = pred.diseae
                class_probs = pred.pred[0].softmax(dim=-1).cpu().numpy()
                class_labels = pred.pred[0].argmax(dim=-1).cpu().numpy()
                class_names = pred.names
                pred.render()  # render bbox in image
                if class_names == 'frog_eye':
                    st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;align:center'><h1 style='color:#000000'>โรคใบจุดตากบ</h1></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[0]}</div>", unsafe_allow_html=True)
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                    # --Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction', use_column_width='always')
                    enter=True
        # if enter==True:
            if class_names == 'frog_eye':
                st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;align:center'><h1 style='color:#000000'>โรคใบจุดตากบ</h1></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[0]}</div>", unsafe_allow_html=True)
            # st.markdown("<div><br></div>", unsafe_allow_html=True)
            elif class_names == 'yellow_leaf':
                st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบเหลือง</h1></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[1]}</div>", unsafe_allow_html=True)
                    # if disease_name == "yellow_leaf":
                    #     st.markdown(description[0], unsafe_allow_html=True)
                    # elif disease_name == "frog_eye":
                    #     st.markdown(description[1], unsafe_allow_html=True)
                    # else:
                    #     st.markdown(description[2], unsafe_allow_html=True)

def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("เลือกเส้นทางที่จะอัปโหลดภาพ", ['จากไฟล์ของเว็บไซต์', 'จากเครื่องของคุณ'])

    # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("หน่วยประมวลผลที่ใช้", ['cpu'], index=1)
    else:
        deviceoption = st.sidebar.radio("หน่วยประมวลผลที่ใช้", ['cpu'], index=0)
    # -- End of Sidebar

    st.header('🌶️ Chili Detection Model')
    st.subheader('⚙️Select options')

    imageInput(deviceoption, datasrc)

if __name__ == '__main__':
    main()


