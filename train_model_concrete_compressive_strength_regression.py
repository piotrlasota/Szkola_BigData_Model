from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# ------------------------------------
# 1) Spark
# ------------------------------------
spark = SparkSession.builder.appName("Concrete-Strength-Regression").getOrCreate()

# ------------------------------------
# 2) Wczytanie danych
#    Podmień ścieżkę na swoją (lokalnie/HDFS/ADLS)
# ------------------------------------
path = "file:///C:/Piotr/szkola/semestr3/big data/lab/concrete+compressive+strength/Concrete_Data.csv"  
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

print("Schema:")
df.printSchema()
print("Rows:", df.count())

# ------------------------------------
# 4) Dobór feature columns (wszystkie numeryczne, bez targetu)
# ------------------------------------
feature_cols = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age",
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Brakuje kolumn cech w CSV: {missing}")

# Rzutowanie na double + drop nulli
for c in feature_cols + ["Strength"]:
    df = df.withColumn(c, col(c).cast("double"))
df = df.dropna(subset=feature_cols + ["Strength"])

# ------------------------------------
# 5) Assembler
# ------------------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# ------------------------------------
# 6) Wybór modelu (ustaw jeden wariant)
# ------------------------------------
MODEL = "rf"  # "lr" / "rf" / "gbt"

if MODEL == "lr":
    reg = LinearRegression(
        featuresCol="features", 
        labelCol="Strength", 
        maxIter=200, 
        regParam=0.0
    )
elif MODEL == "rf":
    reg = RandomForestRegressor(
        featuresCol="features",
        labelCol="Strength",
        numTrees=300,
        maxDepth=12,
        subsamplingRate=0.8,
        seed=42
    )
elif MODEL == "gbt":
    reg = GBTRegressor(
        featuresCol="features",
        labelCol="Strength",
        maxIter=200,
        maxDepth=6,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42
    )
    reg = GBTRegressor(
        featuresCol="features",
        labelCol="Strength",
        maxIter=400,
        stepSize=0.03,
        maxDepth=4,
        subsamplingRate=0.8,
        maxBins=64
    )
else:
    raise ValueError("MODEL musi być jednym z: 'lr', 'rf', 'gbt'")

pipeline = Pipeline(stages=[assembler, reg])

# ------------------------------------
# 7) Podział train/test i trening
# ------------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)

# ------------------------------------
# 8) Predykcja + metryki
# ------------------------------------
pred = model.transform(test)

rmse_eval = RegressionEvaluator(labelCol="Strength", predictionCol="prediction", metricName="rmse")
r2_eval = RegressionEvaluator(labelCol="Strength", predictionCol="prediction", metricName="r2")

rmse = rmse_eval.evaluate(pred)
r2 = r2_eval.evaluate(pred)

print(f"Model: {MODEL}")
print(f"RMSE: {rmse}")
print(f"R2:   {r2}")

pred.select("Strength", "prediction").show(10, truncate=False)

# (opcjonalnie) zapis modelu
model.write().overwrite().save("models/concrete_strength_regression_model")

spark.stop()
