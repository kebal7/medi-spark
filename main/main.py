import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

def create_spark_session():
    return (
        SparkSession.builder.appName("HeartDiseasePredict")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

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

if __name__ == "__main__":
    # Example patient (replace values with real ones)
    test_patient = {
        "age": 54,
        "trestbps": 130,
        "chol": 246,
        "thalach": 150,
        "oldpeak": 1.2,
        "fbs": 0,
        "exang": 0,
        "ca": 0,
        "is_male": 1,
        "cp_typical_angina": 1,
        "cp_atypical_angina": 0,
        "cp_non-anginal": 0,
        "cp_asymptomatic": 0,
        "thal_normal": 1,
        "thal_fixed_defect": 0,
        "thal_reversible_defect": 0,
        "slope_upsloping": 0,
        "slope_flat": 1,
        "slope_downsloping": 0,
    }
    predict(test_patient)

