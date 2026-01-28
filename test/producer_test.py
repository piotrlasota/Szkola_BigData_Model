from kafka import KafkaProducer
import json, uuid

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: k.encode("utf-8"),
)

req_id = f"req-{uuid.uuid4().hex[:8]}"
msg = {
    "request_id": req_id,
    "model": "concrete_strength_v1",
    "payload": {
        "Cement": 540.0,
        "BlastFurnaceSlag": 0.0,
        "FlyAsh": 0.0,
        "Water": 162.0,
        "Superplasticizer": 2.5,
        "CoarseAggregate": 1040.0,
        "FineAggregate": 676.0,
        "Age": 28.0
    }
}

producer.send("concrete.model.request", key=req_id, value=msg)
producer.flush()
print("Sent:", req_id)