import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# -----------------------------#
# 🌿 Custom CSS (Updated)
# -----------------------------#
st.markdown("""
    <style>
        .main {
            background-color: #f0fff0;
        }

        body {
            color: #1b5e20;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1b5e20 !important;
        }

        .stButton>button {
            color: white !important;
            background: linear-gradient(to right, #66bb6a, #388e3c);
            border-radius: 12px;
            font-weight: bold;
        }

        .stTextInput>div>div>input {
            border-radius: 12px;
        }

        .stMarkdown {
            color: #1b5e20;
        }

        section[data-testid="stSidebar"] {
            background-color: #e8f5e9 !important;
        }

        .stSelectbox label, .stSelectbox div {
            color: #1b5e20 !important;
        }

        .css-1vq4p4l, .css-1x8cf1d {
            color: #1b5e20 !important;
        }

        .team-photo {
            height: 300px !important;
            width: 100% !important;
            object-fit: cover !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .stAlert {
            background-color: #fff8e1 !important;
            color: #1b1b1b !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            font-size: 1.1rem !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }

        .stAlert div, .stAlert span, .stAlert p {
            color: #1b1b1b !important;
        }
    </style>
""", unsafe_allow_html=True)

model_path = "trained_model.h5"
file_id = "1bId3SKUybn9y3CKmTo8ZDysrp6oBofeo"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, model_path, quiet=False)



def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("🌿 Plant Disease App")
app_mode = st.sidebar.selectbox(
    "📍 Select Page",
    ["🏠 Home", "📚 About", "🦠 Disease Recognition"],
    index=2
)

if app_mode == "🏠 Home":
    st.header("🌾 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_container_width=True)


    st.markdown("""
    # 🌱 Welcome to the Plant Health Detector!

    Helping farmers, gardeners, and plant lovers identify and tackle plant diseases easily.

    ---

    ## 🚀 What Can You Do?

    - 📷 Diagnose plant issues  
    - ⚡ Get instant predictions  
    - 🌿 Take the right action  

    ---

    ## 🛠️ How to Use

    1. Click on the **🦠 Disease Recognition** tab.  
    2. Upload a clear image of the infected part (leaf/stem).  
    3. Click **Predict** and view results with recommendations.  

    ---

    ## 🌟 Why Choose Us?

    - ✅ Accurate deep learning model  
    - ⚡ Fast results in seconds  
    - 🧑‍💻 Clean and beginner-friendly design  

    ---

    ## 👩‍🔬 Built with 💚 for a healthier planet!

    Want to know about us? Head over to the **📚 About** section!
    """)

elif app_mode == "📚 About":
    st.markdown("## 📖 About This Project")
    st.markdown("""
    This platform uses Deep Learning to detect plant diseases from images.

    **Dataset Used:**
    - 87,000+ RGB leaf images  
    - 38 classes (healthy & diseased)  
    - Trained using Keras CNN on augmented data
    """)

    # st.markdown("## 👩‍👩‍👧 Meet the Team")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.image("Yash.jpeg", caption="Yash Verma", use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
    #     st.markdown("""
    #     **Yash Verma**  
    #     💻 B.Tech CSE  
    #     🔗 [LinkedIn](https://www.linkedin.com/in/yash-verma-b41221241/)  
    #     """, unsafe_allow_html=True)

    # with col2:
    #     st.image("tanisha.jpeg", caption="Tanisha Mahavar",use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
    #     st.markdown("""
    #     **Tanisha Mahavar**  
    #     💻 B.Tech CSE  
    #     🔗 [LinkedIn](https://www.linkedin.com/in/tanisha-mahavar-02ba6b25a/)  
    #     """, unsafe_allow_html=True)

    

    # Apply class via HTML tags for consistent height
    st.markdown("""
        <style>
            img {
                height: 300px !important;
                object-fit: cover !important;
                border-radius: 12px !important;
            }
        </style>
        <h3 style='text-align: center; color: #14532d; background-color: #d1fae5; padding: 10px; border-radius: 10px;'>
            🌱 We aim to combine Agriculture & Technology for a Greener Tomorrow 🌿
        </h3>
    """, unsafe_allow_html=True)

elif app_mode == "🦠 Disease Recognition":
    st.header("Plant Disease Recognition System")
    st.markdown("<h4 style='color:#1b5e20'>📤 Upload Plant Image:</h4>", unsafe_allow_html=True)

    test_image = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_container_width=True)
        else:
            st.warning("⚠️ Please upload an image first!")

    if st.button("Predict"):
        if test_image is not None:
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            with st.spinner("🌿 Analyzing image... Hold tight!"):
                result_index = model_prediction(test_image)
            st.success(f"🧠 Our Prediction: **{class_name[result_index]}**")
        else:
            st.warning("⚠️ Please upload an image before prediction!")
