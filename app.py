import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ১. মডেল লোড করা (এরর হ্যান্ডলিং সহ)
@st.cache_resource
def load_my_model():
    try:
        # সরাসরি .keras ফাইল লোড করার চেষ্টা
        return tf.keras.models.load_model('fabric_defect_model.keras', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

st.title("🛡️ Fabric/Surface Defect Detector")
st.write("ছবি আপলোড করুন এবং দেখুন সেখানে কোনো ডিফেক্ট আছে কি না।")

# ২. ফাইল আপলোডার
uploaded_file = st.file_uploader("একটি ছবি সিলেক্ট করুন...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # ছবি দেখানো
    image = Image.open(uploaded_file)
    st.image(image, caption='আপলোড করা ছবি', use_column_width=True)

    # ৩. ইমেজ প্রি-প্রসেসিং
    img = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ৪. প্রেডিকশন করা
    if st.button('Analyze Surface'):
        prediction = model.predict(img_array)
        result_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader(f"Detection Result: {class_names[result_idx].upper()}")
        st.write(f"Confidence Score: {confidence:.2f}%")

        if confidence > 50:
            st.error("⚠️ Defect Detected!")
        else:
            st.success("✅ Surface Looks Good!")
