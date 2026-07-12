import os

import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "https://insurance-cost-predictor-ctwv.onrender.com/predict",
)

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
    options=["Southwest", "Southeast", "Northwest", "Northeast"],
)

if st.button("Predict the cost"):
    inp_data = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }

    try:
        response = requests.post(API_URL, json=inp_data, timeout=15)
        response.raise_for_status()
        result = response.json()
        prediction = result.get("prediction")

        if prediction is None:
            st.error("The prediction service returned an unexpected response.")
        else:
            st.success(f"Predicted Price: {float(prediction):.2f}")

    except requests.exceptions.Timeout:
        st.error("The prediction service took too long to respond. Please try again shortly.")
    except requests.exceptions.ConnectionError:
        st.error("Unable to reach the prediction service right now. Please try again later.")
    except requests.exceptions.HTTPError as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
        st.error(f"Prediction service error: {detail or str(exc)}")
    except ValueError:
        st.error("The prediction service returned an invalid response.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
