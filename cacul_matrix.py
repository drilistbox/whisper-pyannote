# in this example, we are using AMI first test file.
import os
os.environ["PYANNOTE_DATABASE_CONFIG"] = "database/database.yml"
import torch
from pyannote.audio import Pipeline
from pyannote.database import get_protocol
from pyannote.database import FileFinder
import time

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda:3"))

preprocessors = {'audio': FileFinder()}
dataset = get_protocol('AMI.SpeakerDiarization.MixHeadset',
                        preprocessors=preprocessors)
print('数据集加载完成')
from pyannote.metrics.diarization import DiarizationErrorRate
metric = DiarizationErrorRate()

# time_all = 0
# cnt = 0
for file in dataset.test():
    # apply pretrained pipeline
    # t1 = time.time()
    diarization = pipeline(file)
    # t2 = time.time()
    print(diarization)
    file["pretrained pipeline"] = diarization
    # time_all += (t2-t1)
    # cnt += 1

    # evaluate its performance
    metric(file["annotation"], file["pretrained pipeline"], uem=file["annotated"])
    print(100 * abs(metric))
    # print(f"{(t2-t1):.4f}s")
print(f"The pretrained pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric):.1f}% on {dataset.name} test set.")
# print(f"The average inference time is {time_all/cnt:.4f}s on a GPU A6000")


# for file in dataset.test():
#     diarization = pipeline(file)
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#       print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")