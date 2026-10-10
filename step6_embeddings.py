import numpy as np
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")

documents = [
    "Chip A7 fails timing tests when the supply voltage drops below 0.85V.",
    "To reset the test fixture, power cycle the board and wait 30 seconds.",
    "The weekly yield report is generated every Monday from the test database.",
    "Thermal throttling starts at 95 degrees Celsius to protect the chip.",
    "Calibration of the oscilloscope must be done every six months.",
]

doc_vectors = np.array(list(model.embed(documents)))
print("shape:", doc_vectors.shape)  # (number of documents, size of each vector)

query = "My processor stops working correctly when the power is too low"
query_vector = np.array(list(model.embed([query])))[0]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine(query_vector, v) for v in doc_vectors]

for score, doc in sorted(zip(scores, documents), reverse=True):
    print(f"{score:.3f}  {doc}")