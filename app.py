import streamlit as st
from keras.models import load_model
from PIL import Image

from util import classify, set_background

set_background('bg/bg.png')

# set title
st.title('Tomato Leaf Disease classification')

# set header
st.header('Please upload a Tomato Leaf image')

# upload file 
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model/TomatoLeafMobilenet.h5')

# load class names
label = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
         'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
         'Tomato___healthy']
# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    predicted_class, confidence = classify(image, model, label)

    # write classification
    st.write("## {}".format(predicted_class))
    st.write("### score: {}%".format(int(confidence * 10) / 10))
