from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Load model
model = joblib.load("app/model.joblib")

# FastAPI app
app = FastAPI(title="Student Performance Predictor API")

# Templates & static
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request,
                     hours_studied: float = Form(...),
                     attendance: float = Form(...)):
    X_new = np.array([[hours_studied, attendance]])
    predicted_score = model.predict(X_new)[0]
    predicted_score = round(predicted_score, 2)
    return templates.TemplateResponse("form.html",
                                      {"request": request, "result": predicted_score})

@app.post("/predict")
async def predict_api(hours_studied: float, attendance: float):
    X_new = np.array([[hours_studied, attendance]])
    predicted_score = model.predict(X_new)[0]
    return {"predicted_score": predicted_score}