audio=AudioSegment.from_wav('full_audio.wav')
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# Load the full audio file
audio = AudioSegment.from_wav("full_audio.wav")

# Create output folder
os.makedirs("chunks2", exist_ok=True)

# Smart split: break when silence >= 700ms and volume below -40 dBFS
chunks = split_on_silence(
    audio,
    min_silence_len=400,     # silence must be at least 700ms
    silence_thresh=audio.dBFS - 14,  # threshold relative to audio volume
    keep_silence=100         # keep some silence at the ends
)

# Filter & save: only keep chunks between 2s and 15s
saved = 0
for i, chunk in enumerate(chunks):
    if 4000 <= len(chunk) <= 10000:
        out_file = f"chunks2/chunk_{saved}.wav"
        chunk.export(out_file, format="wav")
        print(f"Saved {out_file} ({len(chunk)/1000:.2f} sec)")
        saved += 1