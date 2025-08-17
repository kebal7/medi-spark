import sys
import os
import psycopg2
from psycopg2 import sql
from pyspark.sql import SparkSession
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.utility import setup_logging, format_time
import time


def create_spark_session(logger):
    logger.debug("Initializing Spark Session with default parameters") 
    return SparkSession.builder \
            .appName("HeartDiseaseDataLoad") \
            .getOrCreate()


def create_postgres_tables(pg_un, pg_pw, logger):
    conn = None
    try:
        conn = psycopg2.connect(
                dbname="postgres",
                user=pg_un,
                password=pg_pw,
                host="localhost",
                port="5432"
        )
        cursor = conn.cursor()
        logger.debug("Successfully connected to postgres database")

        create_table_queries = [
                """
                CREATE TABLE IF NOT EXISTS patient_table (
                    id INTEGER PRIMARY KEY,
                    age INTEGER,
                    sex TEXT,
                    dataset TEXT,
                    cp TEXT,
                    trestbps INTEGER,
                    chol INTEGER,
                    fbs BOOLEAN,
                    restecg TEXT,
                    thalach INTEGER,
                    exang BOOLEAN,
                    oldpeak FLOAT,
                    slope TEXT,
                    ca INTEGER,
                    thal TEXT,
                    num INTEGER
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS patient_feature_table (
                    id INTEGER PRIMARY KEY,
                    age INTEGER,
                    trestbps INTEGER,
                    chol INTEGER,
                    thalach INTEGER,
                    oldpeak FLOAT,
                    fbs BOOLEAN,
                    exang BOOLEAN,
                    ca INTEGER,
                    is_male INTEGER,
                    cp_typical_angina INTEGER,
                    cp_atypical_angina INTEGER,
                    cp_non_anginal INTEGER,
                    cp_asymptomatic INTEGER,
                    thal_normal INTEGER,
                    thal_fixed_defect INTEGER,
                    thal_reversible_defect INTEGER,
                    slope_upsloping INTEGER,
                    slope_flat INTEGER,
                    slope_downsloping INTEGER,
                    num INTEGER
                );
                """
        ]

        for query in create_table_queries:
            cursor.execute(query)
        conn.commit()
        logger.info("PostgreSQL tables created successfully")

    except Exception as e:
        logger.warning(f"Error creating tables: {e}")
    finally:
        logger.debug("Closing connection and cursor to postgres db")
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def load_to_postgres(spark, input_dir, pg_un, pg_pw, logger):
    jdbc_url = "jdbc:postgresql://localhost:5432/postgres"
    connection_properties = {
            "user": pg_un,
            "password": pg_pw,
            "driver": "org.postgresql.Driver"
    }

    tables = [
            ("patient_table", "patient_table"),
            ("patient_feature_table", "patient_feature_table")
    ]

    for parquet_path, table_name in tables:
        try:
            df = spark.read.parquet(os.path.join(input_dir, parquet_path))
            mode = "overwrite"  # for clean loads
            df.write \
                .mode(mode) \
                .jdbc(url=jdbc_url, table=table_name, properties=connection_properties)
            logger.info(f"Loaded {table_name} to PostgreSQL")
        except Exception as e:
            logger.warning(f"Error loading {table_name}: {e}")


if __name__ == "__main__":

    logger = setup_logging("load.log")

    if len(sys.argv) != 4:
        logger.error("Usage: python load/execute.py <input_dir> <pg_un> <pg_pw>")
        sys.exit(1)

    input_dir = sys.argv[1]
    pg_un = sys.argv[2]
    pg_pw = sys.argv[3]

    if not os.path.exists(input_dir):
        logger.error(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    logger.info("Load stage started")
    start = time.time()

    spark = create_spark_session(logger)
    create_postgres_tables(pg_un, pg_pw, logger)
    load_to_postgres(spark, input_dir, pg_un, pg_pw, logger)
    
    end = time.time()
    logger.info("Load stage completed")
    logger.info(f"Total time taken {format_time(end-start)}")

