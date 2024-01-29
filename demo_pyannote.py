from pyannote.audio import Pipeline
from pyannote.audio import Inference
import time

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
t1 = time.time()
diarization = pipeline("audio3.wav")
t2 = time.time()
print('推理时间')
print(t2-t1)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")