from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Heart Disease Predictor")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())

# -----------------------
# Request schema
# -----------------------
class PatientData(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: float
    thalach: float
    exang: bool
    oldpeak: float
    slope: str
    ca: float
    thal: str

# -----------------------
# Spark session
# -----------------------
def create_spark_session():
    return SparkSession.builder.appName("HeartDiseasePredict").config("spark.driver.memory", "2g").getOrCreate()

# -----------------------
# Transform input → one-hot format
# -----------------------
def transform_to_one_hot(data: PatientData):
    row = {
        "age": float(data.age),
        "trestbps": float(data.trestbps),
        "chol": float(data.chol),
        "thalach": float(data.thalach),
        "oldpeak": float(data.oldpeak),
        "fbs": 1 if data.fbs > 120 else 0,
        "exang": 1 if data.exang else 0,
        "ca": float(data.ca),
        "is_male": 1 if data.sex.lower() == "male" else 0,
        # Chest Pain one-hot
        "cp_typical_angina": 1 if data.cp.lower() == "typical angina" else 0,
        "cp_atypical_angina": 1 if data.cp.lower() == "atypical angina" else 0,
        "cp_non-anginal": 1 if data.cp.lower() == "non-anginal" else 0,
        "cp_asymptomatic": 1 if data.cp.lower() == "asymptomatic" else 0,
        # Thalassemia one-hot
        "thal_normal": 1 if data.thal.lower() == "normal" else 0,
        "thal_fixed_defect": 1 if data.thal.lower() == "fixed defect" else 0,
        "thal_reversible_defect": 1 if data.thal.lower() == "reversible defect" else 0,
        # Slope one-hot
        "slope_upsloping": 1 if data.slope.lower() == "upsloping" else 0,
        "slope_flat": 1 if data.slope.lower() == "flat" else 0,
        "slope_downsloping": 1 if data.slope.lower() == "downsloping" else 0,
    }
    return row

# -----------------------
# Prediction function
# -----------------------
def predict(new_data: dict):
    spark = create_spark_session()

    # Path to your saved pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "..", "models")
    lr_model = PipelineModel.load(os.path.join(model_dir, "lr_pipeline"))

    # Turn dict into Spark DataFrame
    df = spark.createDataFrame([new_data])

    # Predict
    preds = lr_model.transform(df)
    preds.select("prediction", "probability").show(truncate=False)

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
async def predict_endpoint(data: PatientData):
    features_dict = transform_to_one_hot(data)
    predict(features_dict)
    return {"message": "Prediction done! Check console for output."}

