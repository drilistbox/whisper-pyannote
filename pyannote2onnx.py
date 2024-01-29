import torch
import onnxruntime as ort
from pyannote.audio import Model
import sys
sys.path.append('/home/liuchen/pyannote-audio')

model = Model.from_pretrained("pyannote/speaker-diarization-3.1/pytorch_model.bin")
print(model)

dummy_input = torch.zeros(3, 1, 32000)
torch.onnx.export(
    model,
    dummy_input,
    "pyannote_new.onnx",
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "B", 1: "C", 2: "T"},
    },
)
print('导出完成')