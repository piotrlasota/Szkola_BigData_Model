from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "concrete.model.request",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
    auto_offset_reset="latest",
    enable_auto_commit=True
)

for m in consumer:
    print("KEY:", m.key, "VALUE:", m.value)
