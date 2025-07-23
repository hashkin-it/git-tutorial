from transformers import pipeline, WhisperProcessor
from google.colab import files
import os
import pandas as pd

# Load processor and forced decoder IDs for Malayalam
processor = WhisperProcessor.from_pretrained("thennal/whisper-medium-ml")
forced_ids = processor.get_decoder_prompt_ids(language="ml", task="transcribe")

# Load ASR model
asr = pipeline(
    "automatic-speech-recognition",
    model="thennal/whisper-medium-ml",
    device=0  # use -1 for CPU
)

# Directory containing your chunk files
chunk_dir = "chunks2"

# Get all .wav files
chunk_files = sorted([
    f for f in os.listdir(chunk_dir) if f.endswith(".wav")
])

# Prepare list to collect transcription data
data = []

# Transcribe each chunk
for file in chunk_files:
    file_path = os.path.join(chunk_dir, file)
    print(f"🔊 Transcribing: {file}")

    result = asr(
        file_path,
        chunk_length_s=30,
        generate_kwargs={"forced_decoder_ids": forced_ids}
    )

    data.append({
        "chunk_name": file,
        "audio_path": file_path,
        "transcription": result["text"]
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("malayalam_transcripts.csv", index=False)


files.download('malayalam_transcripts.csv')