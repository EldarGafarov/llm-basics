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


def chunk_text(text, chunk_size=300, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


chunks = chunk_text(manual, chunk_size=300, overlap=50)

print("document length:", len(manual), "characters")
print("number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

# Did the important sentence survive chunking in one piece?
key_sentence = "Chip A7 is known to fail timing tests when the supply voltage drops below 0.85V."
print("\nkey sentence intact in a single chunk?", any(key_sentence in c for c in chunks))