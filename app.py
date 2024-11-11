import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import requests
from PIL import Image
import io

# Load the pre-trained XGBoost model
model = joblib.load("xgboost_model.joblib")

# Prediction function using the XGBoost model
def predict_house_price(living_area, bathrooms, bedrooms, latitude, longitude):
    data = pd.DataFrame({
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Living Area": [living_area],
        "Bathrooms": [int(bathrooms)],
        "Bedrooms": [int(bedrooms)]
    })
    prediction = model.predict(data)
    return prediction[0]

# Define the AVM endpoint URL
avm_endpoint_url = "http://23.240.69.246:4000/image_predict"  # Replace with your actual AVM endpoint URL

# Streamlit UI layout
st.set_page_config(page_title="House Price Prediction Dashboard", layout="centered")
st.title("House Price Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Input Features")
living_area = st.sidebar.slider("Living Area (sqft)", min_value=1000, max_value=5000, value=1000, step=100)
bathrooms = st.sidebar.selectbox("Bathrooms", list(range(1, 11)), index=1)
bedrooms = st.sidebar.selectbox("Bedrooms", list(range(1, 11)), index=2)
advanced_options = st.sidebar.checkbox("Show Advanced Options")

# Conditional advanced inputs
if advanced_options:
    latitude = st.sidebar.number_input("Latitude", value=37.7749)
    longitude = st.sidebar.number_input("Longitude", value=-122.4194)
else:
    latitude = 37.7749
    longitude = -122.4194

# Perform XGBoost prediction based on current inputs
predicted_price = predict_house_price(living_area, bathrooms, bedrooms, latitude, longitude)

# Section for AVM model prediction with image upload
st.write("### Upload Image of House for AVM Prediction")

# Upload image for AVM model
uploaded_file = st.file_uploader("Choose a house image...", type=["jpg", "jpeg", "png"])
avm_price = "Upload An Image"  # Default placeholder

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_stream = io.BytesIO()
    image.save(image_stream, format=image.format)
    image_stream.seek(0)

    files = {'image': (f'house.{image.format}', image_stream, f'image/{image.format}')}

    # Send the image to the AVM endpoint
    try:
        response = requests.post(
            avm_endpoint_url,
            files=files,
        )
        if response.status_code == 200:
            avm_price = "$" + format(round(response.json().get("prediction", "N/A")[0][0] * 100000, 2), ",")
        else:
            st.write("Error in AVM prediction:", response.status_code, response.text)
    except Exception as e:
        st.write("Error in connecting to AVM endpoint:", str(e))

# Display comparison with AVM
st.write("#### Comparison with AVM Prediction")
comparison_data = pd.DataFrame({
    "Model": ["XGBoost Prediction", "AVM Prediction"],
    "Predicted Price": [f"${predicted_price:,.2f}", f"{avm_price}"]
})
st.table(comparison_data)

st.write("### Baseline Prediction Stats")
# Generate line plot for Living Area vs. Predicted Price using XGBoost
living_area_values = range(500, 5001, 100)
trend_prices = [predict_house_price(area, bathrooms, bedrooms, latitude, longitude) for area in living_area_values]
fig, ax = plt.subplots()
ax.plot(living_area_values, trend_prices, color="blue", linewidth=2)
ax.set_title("Predicted Price by Living Area (XGBoost)")
ax.set_xlabel("Living Area (sqft)")
ax.set_ylabel("Predicted Price ($)")
st.pyplot(fig)

# Generate bar plot for Bedrooms vs. Predicted Price using XGBoost
bedroom_values = range(1, 11)
bedroom_prices = [predict_house_price(living_area, bathrooms, bed, latitude, longitude) for bed in bedroom_values]
fig, ax = plt.subplots()
ax.bar(bedroom_values, bedroom_prices, color="skyblue")
ax.set_title("Predicted Price by Number of Bedrooms (XGBoost)")
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Predicted Price ($)")
ax.set_xticks(bedroom_values)
st.pyplot(fig)

# About section
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard provides dynamic house price predictions based on input values for Living Area, "
    "Longitude, Latitude, Bathrooms, and Bedrooms. Adjust the inputs in the sidebar to see how the "
    "predicted price changes. The AVM model provides additional predictions based on house images."
)