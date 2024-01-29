import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote_whisper_fold.pyannote_whisper.utils import diarize_text
import torch

print("加载声纹模型...")
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda:1"))
output_dir = "./union_out"
print("加载whisper语音模型...")
model = WhisperModel("guillaumekln/faster-whisper-large-v2", device="cuda", compute_type="float16")


def process_audio(file_path):
    print(f"===={file_path}=======")
    diarization_result = pipeline(file_path)
    asr_result, info = model.transcribe(file_path, beam_size=1, word_timestamps=False,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    
    # alignment_model, metadata = whisperx.load_align_model(language_code=info.language, device="cuda")
    # result_aligned = whisperx.align(asr_result, alignment_model, metadata, file_path, "cuda")
    # word_ts = result_aligned["segments"]

    final_result = diarize_text(asr_result, diarization_result)
    output_file = os.path.join(output_dir, os.path.basename(file_path)[:-4] + '.txt')
    with open(output_file, 'w') as f:
        for seg, spk, sent in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}\n'
            f.write(line)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wave_dir = '/home/liuchen/pyannote-audio/wav_files'

# 获取当前目录下所有wav文件名
wav_files = [os.path.join(wave_dir, file) for file in os.listdir(wave_dir)]

# 处理每个wav文件
# with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#     executor.map(process_audio, wav_files)
for wav_file in wav_files:
    process_audio(wav_file)
print('处理完成！')
