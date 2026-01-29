from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, lit, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel

# =========================
# Konfiguracja
# =========================
KAFKA_BOOTSTRAP = "localhost:9092"
REQUEST_TOPIC = "concrete.model.request"
RESPONSE_TOPIC = "concrete.model.response"
CHECKPOINT_DIR = "file:///C:/DEV/tmp/spark_checkpoints/concrete_reqresp"
MODEL_PATH = "models/concrete_strength_regression_model"

EXPECTED_MODEL = "concrete_strength_v1"

feature_cols = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age",
]

# =========================
# Spark
# =========================
spark = (
    SparkSession.builder
    .appName("Concrete-ReqResp-Inference-Worker")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =========================
# Model
# =========================
model = PipelineModel.load(MODEL_PATH)

# =========================
# Schemat JSON request
# =========================
payload_schema = StructType([
    StructField("Cement", DoubleType(), True),
    StructField("BlastFurnaceSlag", DoubleType(), True),
    StructField("FlyAsh", DoubleType(), True),
    StructField("Water", DoubleType(), True),
    StructField("Superplasticizer", DoubleType(), True),
    StructField("CoarseAggregate", DoubleType(), True),
    StructField("FineAggregate", DoubleType(), True),
    StructField("Age", DoubleType(), True),
])

request_schema = StructType([
    StructField("request_id", StringType(), True),
    StructField("model", StringType(), True),
    StructField("payload", payload_schema, True),
])

# =========================
# 1) Read from Kafka (ciągły)
# =========================
raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", REQUEST_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

# value -> JSON (może być zły, wtedy req = null)
parsed = (
    raw.select(
        col("key").cast("string").alias("kafka_key"),
        col("timestamp").alias("kafka_ts"),
        col("value").cast("string").alias("json_str"),
        from_json(col("value").cast("string"), request_schema).alias("req")
    )
)

# rozbij req na kolumny (jeśli req null -> wszystkie będą null)
flat = parsed.select(
    col("kafka_key"),
    col("kafka_ts"),
    col("json_str"),
    col("req.request_id").alias("request_id"),
    col("req.model").alias("model"),
    col("req.payload.*")
)

# =========================
# 2) Walidacja request-response
# =========================

# bad json => req.request_id jest null i w ogóle req null
is_bad_json = col("request_id").isNull() & col("model").isNull() & col("Cement").isNull()

# model check
is_wrong_model = col("model").isNotNull() & (col("model") != lit(EXPECTED_MODEL))

# missing fields check
all_features_present = lit(True)
for c in feature_cols:
    all_features_present = all_features_present & col(c).isNotNull()

is_missing_fields = (
    col("request_id").isNull()
    | col("model").isNull()
    | (~all_features_present)
)

# final validity: musi być poprawny model + komplet pól
is_valid = (~is_bad_json) & (~is_wrong_model) & (~is_missing_fields)

status_col = (
    when(is_bad_json, lit("ERROR_BAD_JSON"))
    .when(is_wrong_model, lit("ERROR_WRONG_MODEL"))
    .when(is_missing_fields, lit("ERROR_MISSING_FIELDS"))
    .otherwise(lit("OK"))
)

# =========================
# 3) Predykcja (tylko valid)
#    Żeby model.transform nie wywalił się na nullach, dajemy prediction tylko gdy valid
# =========================
predicted = model.transform(flat)

with_status = (
    predicted
    .withColumn("status", status_col)
    .withColumn("prediction_out", when(col("status") == "OK", col("prediction")).otherwise(lit(None).cast("double")))
)

# =========================
# 4) Response JSON + Kafka key = request_id
# =========================
# jeśli request_id jest null (bad json), spróbuj użyć kafka_key, a jak też null -> "unknown"
response_key = when(col("request_id").isNotNull(), col("request_id")) \
    .when(col("kafka_key").isNotNull(), col("kafka_key")) \
    .otherwise(lit("unknown"))

response_df = (
    with_status.select(
        response_key.alias("key"),
        to_json(struct(
            when(col("request_id").isNotNull(), col("request_id")).otherwise(response_key).alias("request_id"),
            when(col("model").isNotNull(), col("model")).otherwise(lit(EXPECTED_MODEL)).alias("model"),
            col("prediction_out").alias("prediction"),
            col("status")
        )).alias("value")
    )
    .select(col("key").cast("string"), col("value").cast("string"))
)

# =========================
# 5) Write to Kafka (ciągły)
# =========================
query = (
    response_df.writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("topic", RESPONSE_TOPIC)
    .option("checkpointLocation", CHECKPOINT_DIR)
    .outputMode("append")
    .start()
)

# Trzyma proces “w nieskończoność” (czyli nasłuchuje cały czas)
query.awaitTermination()
