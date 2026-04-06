"""Create minimal test fixtures for integration testing."""
import json
import numpy as np
import soundfile as sf
from pathlib import Path

fixtures = Path("tests/fixtures")
audio_dir = fixtures / "audio" / "test_batch"
audio_dir.mkdir(parents=True, exist_ok=True)

# Create a 2-second silence wav
sr = 16000
silence = np.zeros(sr * 2, dtype=np.float32)
sf.write(str(audio_dir / "test_0000.wav"), silence, sr)

# Create a minimal manifest (matching real format)
manifest = [
    {
        "id": "test_0000",
        "audio": "audio/test_batch/test_0000.wav",
        "text": "test sentence",
        "duration_sec": 2.0,
        "batch": "test_batch",
        "source_id": "test",
        "start_sec": 0.0,
        "end_sec": 2.0,
    }
]
with open(fixtures / "test_manifest.jsonl", "w") as f:
    for item in manifest:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Create a minimal lexicon
with open(fixtures / "test_lexicon.txt", "w") as f:
    f.write("test\n")

print("Fixtures created.")
