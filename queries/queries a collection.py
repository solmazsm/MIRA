import json

results = collection.query(query_texts=["space"], n_results=10)

print(json.dumps(results, indent=4))

collection.query(query_texts=["space"], where={"topic": "SCIENCE"}, n_results=10)
