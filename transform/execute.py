import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F


def create_spark_session():
    """Initialize Spark session."""
    return (
        SparkSession.builder.appName("HeartDiseaseTransform")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )


def load_and_clean(spark, input_dir, output_dir):
    """Load patient CSV, drop duplicates and nulls, save as patient_table."""
    schema = T.StructType([
        T.StructField("id", T.IntegerType(), False),
        T.StructField("age", T.IntegerType(), True),
        T.StructField("sex", T.StringType(), True),
        T.StructField("dataset", T.StringType(), True),
        T.StructField("cp", T.StringType(), True),
        T.StructField("trestbps", T.IntegerType(), True),
        T.StructField("chol", T.IntegerType(), True),
        T.StructField("fbs", T.BooleanType(), True),
        T.StructField("restecg", T.StringType(), True),
        T.StructField("thalach", T.IntegerType(), True),
        T.StructField("exang", T.BooleanType(), True),
        T.StructField("oldpeak", T.FloatType(), True),
        T.StructField("slope", T.StringType(), True),
        T.StructField("ca", T.IntegerType(), True),
        T.StructField("thal", T.StringType(), True),
        T.StructField("num", T.IntegerType(), True)
    ])

    df = spark.read.schema(schema).csv(
        os.path.join(input_dir, "heart_disease_uci.csv"), header=True
    )

    df = df.dropDuplicates(["id"]).na.drop()
    df.write.mode("overwrite").parquet(os.path.join(output_dir, "patient_table"))
    print("Stage 1: patient_table saved")
    return df


def create_feature_table(df, output_dir):
    """Create ML-ready feature table with encoding and scaling."""
    # Encode categorical variables
    df_feat = df.withColumn("is_male", F.when(F.col("sex") == "Male", 1).otherwise(0))

    # One-hot encode 'cp'
    cp_values = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    for val in cp_values:
        df_feat = df_feat.withColumn(f"cp_{val.replace(' ', '_')}", F.when(F.col("cp") == val, 1).otherwise(0))

    # One-hot encode 'thal'
    thal_values = ["normal", "fixed defect", "reversible defect"]
    for val in thal_values:
        df_feat = df_feat.withColumn(f"thal_{val.replace(' ', '_')}", F.when(F.col("thal") == val, 1).otherwise(0))

    # One-hot encode 'slope'
    slope_values = ["upsloping", "flat", "downsloping"]
    for val in slope_values:
        df_feat = df_feat.withColumn(f"slope_{val}", F.when(F.col("slope") == val, 1).otherwise(0))

    # Select numeric and encoded features
    feature_cols = ["id", "age", "trestbps", "chol", "thalach", "oldpeak", "fbs", "exang", "ca", "is_male"] + \
                   [f"cp_{v.replace(' ', '_')}" for v in cp_values] + \
                   [f"thal_{v.replace(' ', '_')}" for v in thal_values] + \
                   [f"slope_{v}" for v in slope_values] + ["num"]

    df_feat = df_feat.select(feature_cols)

    df_feat.write.mode("overwrite").parquet(os.path.join(output_dir, "patient_feature_table"))
    print("Stage 2: patient_feature_table saved")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    spark = create_spark_session()
    patient_df = load_and_clean(spark, input_dir, output_dir)
    create_feature_table(patient_df, output_dir)

    print("Transformation pipeline completed")

