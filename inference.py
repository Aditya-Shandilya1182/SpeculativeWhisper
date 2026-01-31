import time
from dataclasses import dataclass
import torch
import whisper
from jiwer import wer
from src.speculative_decoding import SpeculativeWhisper
from src.utils import load_references, clean_whisper_output



@dataclass
class Config:
    draft_model: str = "tiny"
    final_model: str = "large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    k: int = 5
    beam_search: bool = False
    beam_size: int = 4
    top_p: float | None = 0.9
    max_tokens: int = 200
    mel_dim_tiny: int = 80
    mel_dim_large: int = 128

audio_files = [
    "LibriSpeech/dev-clean/84/121123/84-121123-0000.flac",
    "LibriSpeech/dev-clean/84/121123/84-121123-0001.flac",
    "LibriSpeech/dev-clean/84/121123/84-121123-0002.flac",
    "LibriSpeech/dev-clean/84/121123/84-121123-0003.flac",
    "LibriSpeech/dev-clean/84/121123/84-121123-0004.flac",
]

references = load_references(audio_files)

sw = SpeculativeWhisper(Config())

spec_times = []
spec_outputs = []

for path in audio_files:
    start = time.time()
    out = sw.transcribe([path], max_tokens=100)[0]
    torch.cuda.synchronize()
    t = time.time() - start

    spec_times.append(t)
    spec_outputs.append(out)

spec_preds = [clean_whisper_output(o) for o in spec_outputs]
spec_wer = wer(references, spec_preds)

print("\n--- Speculative Transcription Outputs ---")

for i, out in enumerate(spec_preds):
    print(f"\n[{i}] {out}")

print("\nSpeculative Whisper")
print(f"Avg latency/sample: {sum(spec_times)/len(spec_times):.4f}s")
print(f"Min latency: {min(spec_times):.4f}s")
print(f"Max latency: {max(spec_times):.4f}s")
print(f"Total time: {sum(spec_times):.2f}s")
print(f"WER: {spec_wer:.4f}")

del sw
torch.cuda.empty_cache()
torch.cuda.synchronize()

device = "cuda" if torch.cuda.is_available() else "cpu"
vanilla_model = whisper.load_model("large-v3").to(device)

vanilla_times = []
vanilla_outputs = []

for path in audio_files:
    start = time.time()
    r = vanilla_model.transcribe(path, language="en")
    torch.cuda.synchronize()
    t = time.time() - start

    vanilla_times.append(t)
    vanilla_outputs.append(r["text"])

vanilla_wer = wer(references, vanilla_outputs)

print("\n--- Vanilla Transcription Outputs ---")

for i, out in enumerate(vanilla_outputs):
    print(f"\n[{i}] {out}")

print("\nVanilla Whisper Large-V3")
print(f"Avg latency/sample: {sum(vanilla_times)/len(vanilla_times):.4f}s")
print(f"Min latency: {min(vanilla_times):.4f}s")
print(f"Max latency: {max(vanilla_times):.4f}s")
print(f"Total time: {sum(vanilla_times):.2f}s")
print(f"WER: {vanilla_wer:.4f}")