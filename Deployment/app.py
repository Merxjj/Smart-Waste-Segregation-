import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *

labels = gen_labels()

# Hide Streamlit header & deploy button for a clean UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style = "display: flex; flex-direction: row; align-items: center; justify-content: center;">
     <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here </h3></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
image = None
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
  img_url = st.text_input('Enter the Image Address')
  if img_url:
    try:
      image = load_image_from_url(img_url)
    except Exception as err:
      st.error(f"Please enter a valid direct Image URL! Error: {err}")

try:
  if image is not None:
    st.image(image, width = 300, caption = 'Uploaded Image')
    if st.button('Predict'):
        img = preprocess(image)

        model = model_arc()
        weights_path = "./weights/modelnew.h5"
        if not os.path.exists(weights_path) and os.path.exists("../weights/modelnew.h5"):
            weights_path = "../weights/modelnew.h5"
        model.load_weights(weights_path)

        prediction = model.predict(img[np.newaxis, ...])
        probs = prediction[0].copy()
        
        # Merge cardboard (idx 0) into paper (idx 3)
        probs[3] += probs[0]
        
        # 5 active categories: Glass, Metal, Paper, Plastic, Trash
        active_indices = [1, 2, 3, 4, 5]
        category_names = {1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
        
        best_idx = active_indices[int(np.argmax([probs[i] for i in active_indices]))]
        predicted_label = category_names[best_idx]
        confidence = probs[best_idx] * 100

        st.info(f'Hey! The uploaded image has been classified as "{predicted_label} product" (Confidence: {confidence:.1f}%)')
        
        # 5-Class Confidence Breakdown Chart (Cardboard removed)
        st.subheader("📊 Class Confidence Breakdown")
        prob_dict = {category_names[i].capitalize(): round(float(probs[i]) * 100, 2) for i in active_indices}
        st.bar_chart(prob_dict)
except Exception as e:
  st.info(e)
  pass
