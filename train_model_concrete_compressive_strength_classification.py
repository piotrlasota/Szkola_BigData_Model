from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ------------------------------------
# 1) Spark
# ------------------------------------
spark = SparkSession.builder.appName("Concrete-Strength-Classification").getOrCreate()

# ------------------------------------
# 2) Wczytanie danych
# ------------------------------------
path = "file:///C:/Piotr/Szkola/semestr3/big data/lab/concrete+compressive+strength/Concrete_Data.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

print("Schema:")
df.printSchema()
print("Rows:", df.count())

# ------------------------------------
# 3) Features
# ------------------------------------
feature_cols = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
    "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age",
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Brakuje kolumn cech w CSV: {missing}")

for c in feature_cols + ["Strength"]:
    df = df.withColumn(c, col(c).cast("double"))
df = df.dropna(subset=feature_cols + ["Strength"])

# ------------------------------------
# 4) Klasy EN 206 (tu: Strength jako fck,cyl)
# ------------------------------------
df = df.withColumn(
    "concrete_class",
    when(col("Strength") < 12, lit("below_C12/15"))
    .when(col("Strength") < 16, lit("C12/15"))
    .when(col("Strength") < 20, lit("C16/20"))
    .when(col("Strength") < 25, lit("C20/25"))
    .when(col("Strength") < 30, lit("C25/30"))
    .when(col("Strength") < 35, lit("C30/37"))
    .when(col("Strength") < 40, lit("C35/45"))
    .when(col("Strength") < 45, lit("C40/50"))
    .when(col("Strength") < 50, lit("C45/55"))
    .when(col("Strength") < 55, lit("C50/60"))
    .when(col("Strength") < 60, lit("C55/67"))
    .when(col("Strength") < 70, lit("C60/75"))
    .when(col("Strength") < 80, lit("C70/85"))
    .when(col("Strength") < 90, lit("C80/95"))
    .when(col("Strength") < 100, lit("C90/105"))
    .otherwise(lit("C100/115_plus"))
)

print("Class distribution:")
df.groupBy("concrete_class").count().orderBy(col("concrete_class").asc()).show(truncate=False)

# ------------------------------------
# 5) StringIndexer -> label
#    Fitujemy osobno, żeby mieć labels dla IndexToString
# ------------------------------------
label_indexer = StringIndexer(inputCol="concrete_class", outputCol="label", handleInvalid="keep")
label_model = label_indexer.fit(df)
labels = label_model.labels  # index -> class name

# ------------------------------------
# 6) Assembler
# ------------------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# ------------------------------------
# 7) Model klasyfikacyjny
# ------------------------------------
MODEL = "rf"  # "lr" / "rf"

if MODEL == "lr":
    clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=200, regParam=0.0)
elif MODEL == "rf":
    clf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=300, maxDepth=12, subsamplingRate=0.8, seed=42
    )
else:
    raise ValueError("MODEL musi być jednym z: 'lr', 'rf'")

# Zamiana prediction -> tekst klasy (TU MUSZĄ BYĆ labels!)
to_class = IndexToString(inputCol="prediction", outputCol="predicted_class", labels=labels)

pipeline = Pipeline(stages=[label_model, assembler, clf, to_class])

# ------------------------------------
# 8) Train/test + trening
# ------------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)

# ------------------------------------
# 9) Predykcja + metryki
# ------------------------------------
pred = model.transform(test)

e_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
e_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
e_wp  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
e_wr  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

print(f"Model: {MODEL}")
print(f"Accuracy:          {e_acc.evaluate(pred)}")
print(f"F1 (weighted):     {e_f1.evaluate(pred)}")
print(f"WeightedPrecision: {e_wp.evaluate(pred)}")
print(f"WeightedRecall:    {e_wr.evaluate(pred)}")

pred.select("Strength", "concrete_class", "predicted_class", "probability").show(15, truncate=False)

# ------------------------------------
# 10) Zapis modelu
# ------------------------------------
model.write().overwrite().save("models/concrete_en206_classification_model")
spark.stop()
