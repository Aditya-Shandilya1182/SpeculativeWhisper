import os
import re

def load_references(audio_files):
    refs = []
    for path in audio_files:
        base = os.path.basename(path).replace(".flac", "")
        chapter_dir = os.path.dirname(path)
        speaker_id = os.path.basename(os.path.dirname(chapter_dir))
        chapter_id = os.path.basename(chapter_dir)
        txt_path = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
        with open(txt_path) as f:
            for line in f:
                if line.startswith(base):
                    refs.append(line.strip().split(" ", 1)[1].lower())
                    break
    return refs

def clean_whisper_output(text):
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()
