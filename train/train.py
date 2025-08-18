import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def create_spark_session():
    return (
        SparkSession.builder.appName("HeartDiseaseML")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

def train_model(spark, input_dir):
    # Load ML feature table
    df = spark.read.parquet(os.path.join(input_dir, "patient_feature_table"))

    # Drop missing values (safe training)
    df = df.dropna()

    # Make label binary: 0 = no disease, 1 = disease
    df = df.withColumn("label", (df["num"] > 0).cast("integer"))

    # Feature columns (drop id + num + label)
    feature_cols = [c for c in df.columns if c not in ("id", "num", "label")]

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Train-test split
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    # Logistic Regression
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
    lr_model = lr.fit(train_df)
    lr_preds = lr_model.transform(test_df)

    # Random Forest
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=50)
    rf_model = rf.fit(train_df)
    rf_preds = rf_model.transform(test_df)

    # Evaluators
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    # Logistic Regression evaluation
    lr_auc = evaluator_auc.evaluate(lr_preds)
    lr_acc = evaluator_acc.evaluate(lr_preds)
    print(f"Logistic Regression -> AUC: {lr_auc:.3f}, Accuracy: {lr_acc:.3f}")

    # Random Forest evaluation
    rf_auc = evaluator_auc.evaluate(rf_preds)
    rf_acc = evaluator_acc.evaluate(rf_preds)
    print(f"Random Forest -> AUC: {rf_auc:.3f}, Accuracy: {rf_acc:.3f}")

    # Save models
    # Define path relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # folder where train.py lives
    model_dir = os.path.join(script_dir, "..", "models")     # go one level up and create models folder
    model_dir = os.path.abspath(model_dir)                   # get absolute path
    os.makedirs(model_dir, exist_ok=True)                    # create if not exist

    # Save models
    lr_model.save(os.path.join(model_dir, "lr_model"))
    rf_model.save(os.path.join(model_dir, "rf_model"))

    print(f"Models saved in {model_dir}")

    return lr_model, rf_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <input_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    spark = create_spark_session()
    train_model(spark, input_dir)

