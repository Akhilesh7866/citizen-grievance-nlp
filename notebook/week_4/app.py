from fastapi import FastAPI
from predict import format_output

app = FastAPI()


@app.get("/")
def home():
    return {"message": "🚀 Grievance Model API is running"}


@app.post("/predict")
def predict(text: str):
    return format_output(text)
