1. aktywacja venv
.venv/Scripts/activate

2. uruchomienie sparka i nasluchiwania modelu
python kafka_interface.py


metryki concrete_en206_classification_model
Model: rf
Accuracy:          0.4502923976608187
F1 (weighted):     0.44704104233190106
WeightedPrecision: 0.47525819379366296
WeightedRecall:    0.4502923976608187

Rows: 1030
Class distribution:
+--------------+-----+
|concrete_class|count|
+--------------+-----+
|below_C12/15  |64   |
|C12/15        |77   |
|C16/20        |56   |
|C20/25        |98   |
|C25/30        |103  |
|C30/37        |125  |
|C35/45        |128  |
|C40/50        |109  |
|C45/55        |60   |
|C50/60        |63   |
|C55/67        |53   |
|C60/75        |58   |
|C70/85        |33   |
|C80/95        |3    |
+--------------+-----+