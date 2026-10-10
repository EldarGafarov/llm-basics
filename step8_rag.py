import anthropic
import numpy as np
from fastembed import TextEmbedding

MODEL = "claude-haiku-5-5"
TOP_K = 3

manual = """Chip A7 Test Procedure

Before testing chip A7, make sure the fixture is powered off and the chip is seated correctly. Connect the supply to the VDD pin and set the voltage to 0.90V. Never exceed 1.10V, because higher voltages can permanently damage the chip.

Timing tests must run at three temperatures: 25C, 85C, and 105C. If a timing violation appears at 85C, repeat the test at 25C before reporting a failure. Chip A7 is known to fail timing tests when the supply voltage drops below 0.85V.

Thermal protection starts at 95C. When the chip reaches this temperature it reduces its clock speed automatically. If the chip reaches 110C, the fixture shuts down power and logs a THERMAL_TRIP error.

Lab safety: always wear an anti-static wrist strap when handling chips. Do not place drinks near the test fixtures. If a fixture smells of burning, press the emergency stop button and notify the lab manager immediately.

Japan is a country in East Asia, known for its technology and culture. Many semiconductor equipment suppliers are based there, and several engineers on our team have visited Tokyo and Osaka for training courses.

The oscilloscope must be calibrated every six months. Record the calibration date on the label attached to the device. An oscilloscope that is past its calibration date must not be used for official test results.

The office cafeteria is open from 8:00 to 15:00 on weekdays. Lunch is served at 12:00, and the coffee machine on the second floor is refilled every morning.

Common errors: VDD_LOW means the supply voltage dropped below the allowed range during a test. TIMING_FAIL means the chip missed a timing requirement. THERMAL_TRIP means the chip overheated and the fixture cut the power.

After testing, power off the fixture, wait 30 seconds, and remove the chip. Record all results in the test database before starting the next chip. The weekly yield report is generated every Monday from this database."""


# ---------- Phase 1: prepare (done once) ----------

def chunk_by_paragraph(text, min_chars=80):
    """One chunk per paragraph. A paragraph that is too short (like the title)
    is merged with the one after it."""
    chunks = []
    buffer = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        buffer = buffer + "\n\n" + para if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
chunks = chunk_by_paragraph(manual)
chunk_vectors = np.array(list(embedder.embed(chunks)))
print(f"Indexed {len(chunks)} chunks, vectors shape {chunk_vectors.shape}")


# ---------- Phase 2: answer (done for every question) ----------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question, k=TOP_K):
    """Return the k most similar chunks as (chunk_index, score)."""
    q_vector = np.array(list(embedder.embed([question])))[0]
    scores = [cosine(q_vector, v) for v in chunk_vectors]
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in best]


SYSTEM = (
    "You answer questions about chip testing using ONLY the numbered context chunks provided. "
    "Cite the chunk numbers you used, like [2]. "
    "If the context does not contain the answer, reply exactly: I can't find that in the documents."
)

client = anthropic.Anthropic()


def ask(question):
    hits = retrieve(question)

    print(f"\nQUESTION: {question}")
    print("retrieved:")
    for i, score in hits:
        preview = chunks[i][:60].replace("\n", " ")
        print(f"  chunk {i}  score {score:.3f}  {preview}...")

    # Augmentation: build the prompt from the retrieved chunks
    context = "\n\n".join(f"[{i}] {chunks[i]}" for i, _ in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason != "end_turn":
        print("ANSWER: (incomplete or refused, stop reason:", response.stop_reason + ")")
        return

    answer = next((b.text for b in response.content if b.type == "text"), "")
    print("ANSWER:", answer)
    print("tokens:", response.usage.input_tokens + response.usage.output_tokens)


questions = [
    "At what supply voltage does chip A7 fail timing tests?",
    "What happens if the chip reaches 110C?",
    "When is the cafeteria open?",
    "What can I do in Asia?",
    "Who is the CEO of the company?",
    "What should I do if a timing violation appears at 85C and the chip is also hot?",
]

for q in questions:
    ask(q)