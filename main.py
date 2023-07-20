import pickle
import uuid
from fastapi import FastAPI, Body, status, Request
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import sklearn
from fastapi.templating import Jinja2Templates
import re
import pandas as pd


app = FastAPI()
templates = Jinja2Templates(directory="public")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load('modell.pkl')

def predict(dist, tip, pas):
    test = pd.DataFrame()
    test['trip_distance'] = [dist]
    test['tip_amount'] = [tip]
    test['month'] = [1]
    test['passenger_count'] = [pas]

    pred = model.predict(test)
    return pred[0]



@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
def make_prediction(request: Request, distance: float, tip: float, passengers: int):
    prediction = predict(distance, tip, passengers)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})