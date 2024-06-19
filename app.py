import streamlit as st
import torch
from PIL import Image
from datetime import datetime
import os
import pathlib
from fastai.vision.all import load_learner

def load_models():
    GOOD_OR_BAD = pathlib.Path("models/good_or_bad.pkl")
    good_or_bad = load_learner(GOOD_OR_BAD)
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', source='local')
    return good_or_bad, detection_model

def predict_image(good_or_bad, detection_model, imgpath, outputpath):
    checkleaf = good_or_bad.predict(imgpath)
    if checkleaf[0] == 'good':
        detection_model.cpu()
        pred = detection_model(imgpath)
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)
        return True, outputpath
    else:
        return False, None

def display_predictions(col2, outputpath, description):
    img_ = Image.open(outputpath)
    with col2:
        st.image(img_, caption='Model Prediction', use_column_width='always')
    st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบจุดตากบ</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[0]}</div>", unsafe_allow_html=True)
    st.markdown("<div><br></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบเหลือง</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[1]}</div>", unsafe_allow_html=True)

def imageInput(device, src, good_or_bad, detection_model):
    enter = False
    description = [
        """โรคใบจุดตากบ (frog_eye) จะพบประจำในพริกและยาสูบ จุดแผลจะกลม กลางแผลมีสีเทาขอบแผลสีน้ำตาล แผลจะกระจายทั่วไป โรคนี้เกิดจากเชื้อเซอโคสะปอร่า (Cercospora) การควบคุมโรคนี้ใช้สารชนิดเดียวกับที่ใช้ควบคุมแอนแทรคโนสก็ได้ หรือจะใช้สารประเภทคลอโรธาโรนิล (chlorothalonil) ฉีดพ่นสม่ำเสมอขณะระบาด จะได้ผลดี""",
        """โรคใบเหลือง (yellow_leaf) สาเหตุส่วนใหญ่ที่ทำให้ต้นไม้ใบเหลือง เป็นเพราะรดน้ำมากเกินไป ความชื้นในดินสูง หรือดินแน่น ระบายน้ำยาก วิธีแก้ปัญหาเบื้องต้นคือ ตัดใบส่วนนั้นทิ้งไป จากนั้นเว้นการรดน้ำไปสักระยะ แล้วค่อยกลับมารดน้ำใหม่เมื่อดินแห้ง เช็กง่าย ๆ โดยใช้นิ้วกดลงไปในดินประมาณ 1 นิ้ว หากหน้าดินแห้งก็รดน้ำได้ แต่ถ้าดินยังแฉะก็ควรรอก่อน"""
    ]

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
            success, outputpath = predict_image(good_or_bad, detection_model, imgpath, outputpath)
            if success:
                display_predictions(col2, outputpath, description)
            else:
                st.error('ไม่สามารถใช้ภาพนี้ได้ กรุณาอัปโหลดใหม่อีกครั้ง')
    elif src == 'จากไฟล์ของเว็บไซต์':
        test_images = os.listdir('data/images/')
        test_image = st.selectbox('เลือกภาพของเว็บไซต์:', test_images)
        image_file = 'data/images/' + test_image
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        if image_file is not None and submit:
            success, outputpath = predict_image(good_or_bad, detection_model, image_file, os.path.join('data/outputs', os.path.basename(image_file)))
            if success:
                display_predictions(col2, outputpath, description)
                enter = True
        if enter:
            st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;align:center'><h1 style='color:#000000'>โรคใบจุดตากบ</h1></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[0]}</div>", unsafe_allow_html=True)
            st.markdown("<div><br></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#FF6F61;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000000'>โรคใบเหลือง</h1></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6F61'>{description[1]}</div>", unsafe.allow_html=True)

def main():
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("เลือกเส้นทางที่จะอัปโหลดภาพ", ['จากไฟล์ของเว็บไซต์', 'จากเครื่องของคุณ'])

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("หน่วยประมวลผลที่ใช้", ['cpu', 'cuda'], index=1)
    else:
        deviceoption = st.sidebar.radio("หน่วยประมวลผลที่ใช้", ['cpu'], index=0)

    st.header('🌶️ Chili Detection Model')
    st.subheader('⚙️Select options')

    good_or_bad, detection_model = load_models()
    imageInput(deviceoption, datasrc, good_or_bad, detection_model)

if __name__ == '__main__':
    main()
