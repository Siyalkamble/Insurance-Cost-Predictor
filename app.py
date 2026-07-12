from fastapi import FastAPI, HTTPException
from schema.user_input import UserInput
from model.predict import predict_output, MODEL_VERSION

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to Insurance Price Predictor API"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(data: UserInput):
    try:
        user_input = {
            "age": data.age,
            "sex": data.sex,
            "bmi": data.bmi,
            "children": data.children,
            "smoker": data.smoker,
            "region": data.region
        }

        prediction = predict_output(user_input)

        return {"prediction": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))