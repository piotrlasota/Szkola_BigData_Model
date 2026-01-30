from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, lit, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

# =========================
# Konfiguracja
# =========================
KAFKA_BOOTSTRAP = "localhost:9092"
REQUEST_TOPIC = "concrete.model.request"
RESPONSE_TOPIC = "concrete.model.response"

# ŚCIEŻKA DO MODELU
MODEL_PATH = "models/concrete_en206_classification_model"

# Ten worker obsługuje TYLKO taki model
EXPECTED_MODEL = "concrete_strength_rf_classification"

# !!! ważne: unikalny checkpoint per worker / per model
CHECKPOINT_DIR = "file:///C:/DEV/tmp/spark_checkpoints/reqresp_concrete_strength_rf_classification"

feature_cols = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age",
]

# =========================
# Spark
# =========================
spark = (
    SparkSession.builder
    .appName(f"Concrete-ReqResp-Worker-{EXPECTED_MODEL}")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =========================
# Model
# =========================
model = PipelineModel.load(MODEL_PATH)
print("Loaded model stages:", [type(s).__name__ for s in model.stages])

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

# value -> JSON (jeśli nie pasuje do schematu, req będzie null)
parsed = raw.select(
    col("key").cast("string").alias("kafka_key"),
    col("timestamp").alias("kafka_ts"),
    col("value").cast("string").alias("json_str"),
    from_json(col("value").cast("string"), request_schema).alias("req"),
)

flat = parsed.select(
    col("kafka_key"),
    col("kafka_ts"),
    col("json_str"),
    col("req.request_id").alias("request_id"),
    col("req.model").alias("model"),
    col("req.payload.*"),
)

# =========================
# 2) HARD FILTER: tylko mój model
#    (wszystko inne ignorujemy -> nic nie wyślemy na response)
# =========================
only_mine = flat.filter(col("model") == lit(EXPECTED_MODEL))

# =========================
# 3) Filtr na komplet danych
#    (braki -> ignorujemy, nie odsyłamy nic)
# =========================
all_present = lit(True)
for c in feature_cols:
    all_present = all_present & col(c).isNotNull()

valid_df = only_mine.filter(col("request_id").isNotNull() & all_present)

# =========================
# 4) Predykcja (TYLKO dla valid_df)
# =========================
predicted = model.transform(valid_df)

# predicted_class jeśli istnieje (IndexToString), inaczej prediction jako string
prediction_text_col = when(col("predicted_class").isNotNull(), col("predicted_class")) \
    .otherwise(col("prediction").cast("string"))

with_out = (
    predicted
    .withColumn("prediction_out", prediction_text_col)
    .withColumn("probability_out", to_json(vector_to_array(col("probability"))))
)

# =========================
# 5) Response JSON + Kafka key = request_id
# =========================
response_df = (
    with_out.select(
        col("request_id").alias("key"),
        to_json(struct(
            col("request_id").alias("request_id"),
            col("model").alias("model"),
            col("prediction_out").alias("prediction"),
            col("probability_out").alias("probability"),
            lit("OK").alias("status"),
        )).alias("value"),
    )
    .select(col("key").cast("string"), col("value").cast("string"))
)

# =========================
# 6) Write to Kafka (ciągły)
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

query.awaitTermination()
