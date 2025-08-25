from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import random  # just for example
import json

app = FastAPI()

# Serve static files (HTML/JS)
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    # Example: random prediction, replace with PySpark model logic
    prediction = random.choice([0, 1])
    probability = random.random()
    return JSONResponse({"prediction": prediction, "probability": probability})

