# SpeculativeWhisper

Implementation of **Speculative Decoding** for the Whisper ASR model.

---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/Aditya-Shandilya1182/SpeculativeWhisper.git
2. 
   ```bash
   cd SpeculativeWhisper
3. Install dependencies
   ```bash
   pip install -r requirements.txt
## Usage
```python
import torch
from dataclasses import dataclass
from speculative_whisper.speculative_decoding import SpeculativeWhisper

@dataclass
class Config:
    draft_model: str = "tiny"
    final_model: str = "large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    k: int = 5
    beam_search: bool = True
    beam_size: int = 4
    top_p: float | None = 0.9
    max_tokens: int = 200
    mel_dim_tiny: int = 80
    mel_dim_large: int = 128

audio_files = ["audio1.flac", "audio2.flac"]

sw = SpeculativeWhisper(Config())
outputs = sw.transcribe(audio_files)

for i, out in enumerate(outputs):
    print(f"\n[{i}] {out}")
```
## For Notebook
1. Open the notebook in Google Colab
2. Update the Config class according to your requirements
3. Click Run All

## Results

The table below compares **Speculative Whisper** (using a *tiny* draft model and *large-v3* final model) against **Vanilla Whisper (large-v3)** on **5 LibriSpeech samples**.  
Speculative Whisper is evaluated with **beam search enabled** and **top-p = 0.9**.

| Metric               | Vanilla Whisper (Large-V3) | Speculative Whisper |
|----------------------|----------------------------|---------------------|
| Avg latency / sample | 1.8475 s                   | 3.2864 s            |
| Min latency          | 1.0581 s                   | 1.9662 s            |
| Max latency          | 2.5962 s                   | 5.6525 s            |
| Total time           | 9.24 s                     | 16.43 s             |
| WER                  | 0.1237                     | 0.1237              |
