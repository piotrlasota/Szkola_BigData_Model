1. aktywacja venv
.venv/Scripts/activate

2. uruchomienie sparka i nasluchiwania modelu
python kafka_interface.py


metryki concrete_en206_classification_model
Model: rf
Accuracy:          0.5146198830409356
F1 (weighted):     0.5152663289425821
WeightedPrecision: 0.530740044555834
WeightedRecall:    0.5146198830409356

Rows: 1030
Class distribution:
+--------------+-----+
|concrete_class|count|
+--------------+-----+
|C12/15        |77   |
|C16/20        |56   |
|C20/25        |98   |
|C25/30        |103  |
|C30/37        |125  |
|C35/45        |128  |
|C40/50        |109  |
|C45/55        |60   |
|C50/60_plus   |210  |
|below_C12/15  |64   |
+--------------+-----+