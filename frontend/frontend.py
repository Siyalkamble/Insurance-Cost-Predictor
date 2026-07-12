import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Medical Insurance Price Predictor")

st.title("Medical Insurance Price Predictor")
st.markdown("Enter your details below")

age = st.number_input("Age", min_value=1, max_value=149, value=30, step=1)
sex = st.selectbox("Gender", options=["Male", "Female"])
bmi = st.number_input("BMI", min_value=1.0, max_value=49.9, value=30.0, step=0.1)
children = st.number_input("No. of Children", min_value=0, value=0, step=1)
smoker = st.selectbox("Are you a smoker?", options=["Yes", "No"])
region = st.selectbox(
    "Region",
    options=["Southwest", "Southeast", "Northwest", "Northeast"]
)

if st.button("Predict the cost"):
    inp_data = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }

    try:
        response = requests.post(API_URL, json=inp_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Price: {result['prediction']}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure the backend is running on port 8000.")

    except requests.exceptions.Timeout:
        st.error("The request timed out. The backend took too long to respond.")

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")