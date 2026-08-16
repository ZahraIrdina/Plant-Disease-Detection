import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# Load trained model
model = load_model("plant_model.keras")

# Load class indices from JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse mapping: index to class label
index_to_label = {v: k for k, v in class_indices.items()}

# Recommendations for each disease
def get_recommendations():
    recommendations = {
        "Pepper__bell___Bacterial_spot": (
            "🟢 *Recommendation:* Remove and destroy infected plants. "
            "Avoid overhead irrigation to reduce leaf wetness. "
            "Apply copper-based bactericides early and regularly."
        ),
        "Pepper__bell___healthy": (
            "✅ Your pepper plant is healthy. "
            "Continue monitoring and maintain good air circulation. "
            "Avoid overwatering and ensure proper nutrient balance."
        ),
        "Potato___Early_blight": (
            "🟢 *Recommendation:* Use fungicides like chlorothalonil or mancozeb. "
            "Practice crop rotation and remove infected debris. "
            "Maintain proper spacing to improve air circulation."
        ),
        "Potato___healthy": (
            "✅ Your potato plant is healthy. "
            "Ensure well-drained soil and monitor regularly for early signs of disease."
        ),
        "Potato___Late_blight": (
            "🟢 *Recommendation:* Apply fungicides like metalaxyl or chlorothalonil. "
            "Destroy infected plants immediately. "
            "Avoid wet foliage and use certified disease-free seed potatoes."
        ),
        "Tomato_Bacterial_spot": (
            "🟢 *Recommendation:* Remove affected leaves and avoid working with wet plants. "
            "Apply copper-based sprays and rotate crops annually. "
            "Use disease-resistant tomato varieties if available."
        ),
        "Tomato_Early_blight": (
            "🟢 *Recommendation:* Use fungicides such as chlorothalonil or mancozeb. "
            "Prune lower leaves and provide staking to avoid soil contact. "
            "Rotate crops and clean up plant debris after harvest."
        ),
        "Tomato_healthy": (
            "✅ Your tomato plant is healthy. "
            "Water at the base to avoid wetting the leaves. "
            "Fertilize appropriately and inspect weekly for signs of infection."
        ),
        "Tomato_Late_blight": (
            "🟢 *Recommendation:* Remove and destroy infected plants. "
            "Apply systemic fungicides like metalaxyl. "
            "Avoid overcrowding and ensure proper drainage in the soil."
        )
    }
    return recommendations

# Streamlit interface
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the disease and get treatment recommendations.")

uploaded_file = st.file_uploader("📁 Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼 Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_label[predicted_index]  # ✅ Use mapping from index to label

    # Display result
    st.success(f"🔍 Prediction: {predicted_class}")

    # Show recommendation
    recommendations = get_recommendations()
    if predicted_class in recommendations:
        st.info(recommendations[predicted_class])
    else:
        st.warning("⚠️ No recommendation available for this class.")