from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import log1p, col

# 1. Spark session
spark = SparkSession.builder \
    .appName("OnlineNewsPopularity_RF_Regression") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

# 2. Wczytanie danych
df = spark.read.csv(
    "file:///C:/Users/piotr/OneDrive/Pulpit/Piotr/szkola/semestr3/big data/lab/OnlineNewsPopularity/OnlineNewsPopularity.csv",
    header=True,
    inferSchema=True,
    ignoreLeadingWhiteSpace=True,
    ignoreTrailingWhiteSpace=True
)
df = df.toDF(*[c.strip() for c in df.columns])  # nadal warto

# 3. Usunięcie kolumny URL
df = df.drop("url")

# Transformacja logarytmiczna zmiennej docelowej w celu redukcji skośności rozkładu
df = df.withColumn("shares", log1p(col("shares")))

# 4. Zmienna docelowa
label_col = "shares"

# 5. Przygotowanie cech
feature_cols = [c for c in df.columns if c != label_col]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

# 6. Model regresyjny Random Forest
rf = RandomForestRegressor(
    featuresCol="scaledFeatures",
    labelCol=label_col,
    numTrees=100,
    maxDepth=10,
    seed=42
)

# 7. Pipeline
pipeline = Pipeline(stages=[
    assembler,
    scaler,
    rf
])

# 8. Podział danych
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 9. Trenowanie modelu
model = pipeline.fit(train_df)

# 10. Predykcja
predictions = model.transform(test_df)

# 11. Ewaluacja
rmse_evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="r2"
)

rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# 12. Zapis modelu (do użycia w Spark Streaming)
model.write().overwrite().save("models/online_news_rf_regression")
