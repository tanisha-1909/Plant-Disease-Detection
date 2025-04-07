import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

st.markdown("""
    <style>
    .footer {
        position: fixed !important;
        left: 0 !important;
        bottom: 0 !important;
        width: 100% !important;
        background-color: #a5d6a7 !important;
        color: #1b5e20 !important;
        text-align: center !important;
        padding: 10px !important;
        font-weight: bold !important;
        border-top: 2px solid #1b5e20 !important;
        z-index: 999 !important;
    }
    </style>
    <div class="footer">
        Made with 💚 by Team GreenTech 
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .corner-ribbon {
        width: 150px;
        background: #1b5e20;
        color: #fff;
        text-align: center;
        font-weight: bold;
        line-height: 25px;
        transform: rotate(-45deg);
        position: fixed;
        top: 25px;
        left: -45px;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        font-size: 12px;
    }
    </style>
    <div class="corner-ribbon">By Tanisha Mahavar</div>
""", unsafe_allow_html=True)


# -----------------------------#
# 🌿 Custom CSS (Updated)
# -----------------------------#
st.markdown("""
    <style>
        .main {
            background-color: #f0fff0 !important;
        }

        body {
            background-color: #f0fff0 !important;
            color: #1b5e20 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1b5e20 !important;
        }

        .stButton>button {
            color: white !important;
            background: linear-gradient(to right, #66bb6a, #388e3c) !important;
            border-radius: 12px;
            font-weight: bold;
        }

        .stTextInput>div>div>input {
            border-radius: 12px !important;
        }

        .stMarkdown {
            color: #1b5e20 !important;
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }

        .stAlert {
            background-color: #fff8e1 !important;
            color: #1b1b1b !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            font-size: 1.1rem !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.1) !important;
        }

        .stAlert div, .stAlert span, .stAlert p {
            color: #1b1b1b !important;
        }
    </style>
""", unsafe_allow_html=True)


url = "https://drive.google.com/file/d/19miyAapKU1saabVDR050q_Qtyz2Guq_s/view?usp=sharing"
model_path = "trained_model.h5"

# Only download if the model is not already downloaded
if not os.path.exists(model_path):
    print("Model not found locally. Downloading...")
    gdown.download(url=url, output=model_path, fuzzy=True)
else:
    print("Model already exists. Skipping download.")



def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("🌿 Plant Disease Detection WebApp")
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

    st.markdown("## 👩‍👩‍👧 Meet the Team")

    col1, col2 ,col3,col4= st.columns(4)

    with col1:
        st.image("Yash.jpeg", caption="Yash Verma", use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
        st.markdown("""
        **Yash Verma**  
        💻 B.Tech CSE  
        🔗 [LinkedIn](https://www.linkedin.com/in/yash-verma-b41221241/)  
        """, unsafe_allow_html=True)

    with col2:
        st.image("tanishamahavar.jpeg", caption="Tanisha Mahavar",use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
        st.markdown("""
        **Tanisha Mahavar**  
        💻 B.Tech CSE  
        🔗 [LinkedIn](https://www.linkedin.com/in/tanisha-mahavar-02ba6b25a/)  
        """, unsafe_allow_html=True)

    with col3:
        st.image("tanishabansal.jpeg", caption="Tanisha Bansal",use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
        st.markdown("""
        **Tanisha Bansal**  
        💻 B.Tech CSE  
        🔗 [LinkedIn](https://www.linkedin.com/in/tanisha-bansal-762b7b255/)  
        """, unsafe_allow_html=True)

    with col4:
        st.image("vidit.jpeg", caption="Vidit Shandilya",use_container_width=True, output_format="JPEG", channels="RGB", clamp=False)
        st.markdown("""
        **Vidit Shandilya**  
        💻 B.Tech CSE  
        🔗 [LinkedIn](https://www.linkedin.com/in/vidit-shandilya-b104b6256/)  
        """, unsafe_allow_html=True)

    

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
                predicted_class = class_name[result_index]
            st.success(f"🧠 Our Prediction: **{class_name[result_index]}**")

            disease_info = {
                    "Apple___Apple_scab": {
                        "description": "Apple scab is a fungal disease causing dark, scabby lesions on leaves and fruit.",
                        "remedy": "Apply fungicides during early leaf development and ensure proper pruning for air circulation."
                    },
                    "Apple___Black_rot": {
                        "description": "Black rot causes circular, black spots on leaves and fruit decay.",
                        "remedy": "Remove and destroy infected fruit; apply copper-based sprays."
                    },
                    "Apple___Cedar_apple_rust": {
                        "description": "This disease alternates between apple and cedar trees, producing bright orange spots.",
                        "remedy": "Use resistant varieties and fungicidal sprays during early spring."
                    },
                    "Apple___healthy": {
                        "description": "The apple leaf is healthy with no signs of infection.",
                        "remedy": "No remedy needed. Maintain good orchard hygiene."
                    },
                    "Blueberry___healthy": {
                        "description": "The blueberry plant is healthy and free from disease.",
                        "remedy": "Maintain proper soil pH and irrigation practices."
                    },
                    "Cherry_(including_sour)___Powdery_mildew": {
                        "description": "Powdery mildew causes white fungal growth on cherry leaves.",
                        "remedy": "Prune trees for air circulation and apply sulfur-based fungicides."
                    },
                    "Cherry_(including_sour)___healthy": {
                        "description": "Cherry plant is healthy and disease-free.",
                        "remedy": "No treatment necessary, continue regular maintenance."
                    },
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
                        "description": "This fungal disease causes rectangular grayish lesions on corn leaves.",
                        "remedy": "Rotate crops, remove debris, and apply fungicides when needed."
                    },
                    "Corn_(maize)___Common_rust_": {
                        "description": "Common rust creates reddish-brown pustules on leaves.",
                        "remedy": "Plant resistant hybrids and apply fungicides if severe."
                    },
                    "Corn_(maize)___Northern_Leaf_Blight": {
                        "description": "It leads to long, gray-green lesions on leaves reducing photosynthesis.",
                        "remedy": "Use resistant varieties and apply foliar fungicides early."
                    },
                    "Corn_(maize)___healthy": {
                        "description": "The corn plant is healthy with no visible disease symptoms.",
                        "remedy": "Ensure adequate watering and nitrogen levels."
                    },
                    "Grape___Black_rot": {
                        "description": "Black rot is a fungal disease causing shriveled, black fruit.",
                        "remedy": "Remove mummified berries and apply early-season fungicides."
                    },
                    "Grape___Esca_(Black_Measles)": {
                        "description": "Esca causes dark streaks in the wood and tiger-striped leaves.",
                        "remedy": "Avoid pruning in wet weather and remove affected vines."
                    },
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
                        "description": "Characterized by brown angular spots leading to premature defoliation.",
                        "remedy": "Use fungicides and improve air flow in vineyards."
                    },
                    "Grape___healthy": {
                        "description": "The grape vine is in good health and shows no signs of disease.",
                        "remedy": "Continue proper vineyard management practices."
                    },
                    "Orange___Haunglongbing_(Citrus_greening)": {
                        "description": "A bacterial disease spread by psyllids, causing yellow shoots and misshapen fruits.",
                        "remedy": "Remove infected trees and control psyllid population with insecticides."
                    },
                    "Peach___Bacterial_spot": {
                        "description": "Causes dark, sunken spots on leaves and fruit, common in wet conditions.",
                        "remedy": "Use copper sprays and plant resistant varieties."
                    },
                    "Peach___healthy": {
                        "description": "The peach tree appears healthy and disease-free.",
                        "remedy": "No remedy needed; maintain optimal growing conditions."
                    },
                    "Pepper,_bell___Bacterial_spot": {
                        "description": "Water-soaked lesions on leaves and fruit that turn brown and dry.",
                        "remedy": "Use certified seeds and copper-based bactericides."
                    },
                    "Pepper,_bell___healthy": {
                        "description": "Bell pepper plant is thriving and shows no disease symptoms.",
                        "remedy": "Ensure consistent watering and nutrient supply."
                    },
                    "Potato___Early_blight": {
                        "description": "Dark concentric spots on leaves often with yellow halos.",
                        "remedy": "Use crop rotation and fungicides like chlorothalonil."
                    },
                    "Potato___Late_blight": {
                        "description": "Causes water-soaked lesions leading to leaf death and tuber rot.",
                        "remedy": "Use resistant varieties and systemic fungicides."
                    },
                    "Potato___healthy": {
                        "description": "Potato foliage is green and disease-free.",
                        "remedy": "Maintain good soil drainage and monitor regularly."
                    },
                    "Raspberry___healthy": {
                        "description": "The raspberry plant is healthy with no fungal or bacterial infections.",
                        "remedy": "No remedy needed. Monitor regularly for pests."
                    },
                    "Soybean___healthy": {
                        "description": "Soybean crop shows no disease symptoms.",
                        "remedy": "Continue crop rotation and weed control practices."
                    },
                    "Squash___Powdery_mildew": {
                        "description": "White, powdery fungal growth on leaves that may cause distortion.",
                        "remedy": "Use neem oil or sulfur sprays and improve air circulation."
                    },
                    "Strawberry___Leaf_scorch": {
                        "description": "Small, reddish-purple spots that merge, causing leaf scorch.",
                        "remedy": "Remove infected leaves and apply protective fungicides."
                    },
                    "Strawberry___healthy": {
                        "description": "The strawberry plant is healthy and fruiting properly.",
                        "remedy": "Maintain moisture levels and fertilize appropriately."
                    },
                    "Tomato___Bacterial_spot": {
                        "description": "Small dark spots on leaves and fruit which may drop prematurely.",
                        "remedy": "Use resistant seeds and copper-based sprays."
                    },
                    "Tomato___Early_blight": {
                        "description": "Concentric rings on older leaves leading to yellowing and drop.",
                        "remedy": "Prune infected leaves and rotate crops annually."
                    },
                    "Tomato___Late_blight": {
                        "description": "Dark blotches on leaves and fruit with white mold underneath.",
                        "remedy": "Apply systemic fungicides like metalaxyl early."
                    },
                    "Tomato___Leaf_Mold": {
                        "description": "Yellow patches on upper side of leaves with fuzzy mold below.",
                        "remedy": "Ensure good airflow and apply chlorothalonil-based fungicides."
                    },
                    "Tomato___Septoria_leaf_spot": {
                        "description": "Numerous small spots with gray centers and dark edges.",
                        "remedy": "Remove infected leaves and use preventive fungicides."
                    },
                    "Tomato___Spider_mites Two-spotted_spider_mite": {
                        "description": "Causes stippling on leaves and webbing under severe infestation.",
                        "remedy": "Use insecticidal soap or neem oil and keep plants hydrated."
                    },
                    "Tomato___Target_Spot": {
                        "description": "Dark, sunken spots with concentric rings on leaves and fruits.",
                        "remedy": "Improve field sanitation and apply fungicides as needed."
                    },
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
                        "description": "Leads to yellowing, curling leaves and stunted plant growth.",
                        "remedy": "Use resistant varieties and control whitefly vectors."
                    },
                    "Tomato___Tomato_mosaic_virus": {
                        "description": "Mottled, distorted leaves with stunted plant development.",
                        "remedy": "Avoid handling plants when wet; disinfect tools regularly."
                    },
                    "Tomato___healthy": {
                        "description": "The tomato plant is lush and free from any known diseases.",
                        "remedy": "Continue proper watering, staking, and fertilization."
                    }
                }

            if predicted_class in disease_info:
                st.subheader("🩺 Disease Description")
                st.info(disease_info[predicted_class]["description"])

                st.subheader("🌿 Suggested Remedy")
                st.warning(disease_info[predicted_class]["remedy"])
            else:
                st.info("ℹ️ No additional info available for this disease.")
        else:
            st.warning("⚠️ Please upload an image before prediction!")

# added remedies
