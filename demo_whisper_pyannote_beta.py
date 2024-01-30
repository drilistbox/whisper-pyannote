import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote_whisper_fold.pyannote_whisper.utils import diarize_text
import torch
import time
# import whisperx

wave_dir = '/data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/demo'
output_dir = '/data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/out_results/'
is_merge = False
if not os.path.exists(output_dir): os.mkdir(output_dir)

print("加载声纹模型...")
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))
print("加载whisper语音模型...")
model = WhisperModel("guillaumekln/faster-whisper-large-v2", device="cuda", compute_type="float16")

'''
# 01 download video
import yt_dlp
# yt_url = 'https://www.youtube.com/watch?v=tgdJkAx3fJM'
# speakers = ["鍋島さん","ハルクさん"]
# yt_url = 'https://www.youtube.com/watch?v=ChbJfLSI5AE'
# speakers = ["鍋島さん","ハルクさん"]
# yt_url = 'https://www.youtube.com/watch?v=dn8Bs1eXQ8o'
# speakers = [r"郭麒麟",r"黄磊", r"主持人", r"周迅", r"岳云鹏"]
yt_url = 'https://www.youtube.com/watch?v=Hi0Fp_nZSZ0'
speakers = [r"何炅", r"黄磊", r"谢娜", r"白百何", r"赵丽颖", r"恩泰"]
ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'outtmpl': '%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(yt_url)
    video_info = ydl.extract_info(yt_url, download=False)
    file_name = f"{video_info['id']}.wav"

# 02 preprocess
import demucs.separate
import shlex
file_extension = ['.mp4', '.wav']
allowed_files = [file for file in os.listdir() if any(file.lower().endswith(ext) for ext in file_extension)]
input_file = max(allowed_files, key=lambda file: os.path.getctime(file))
demucs.separate.main(shlex.split(f'-n htdemucs --two-stems=vocals "{input_file}" -o "temp_outputs"'))
input_file = os.path.join(
        "temp_outputs", "htdemucs", os.path.basename(input_file[:-4]), "vocals.wav")
audio_file = "audio_16k.wav"
os.system("rm -rf {}".format(audio_file))
print("rm -rf {}".format(audio_file))
os.system("ffmpeg -i {} -ac 1 -ar 16000 {}".format(input_file, audio_file))
print("ffmpeg -i {} -ac 1 -ar 16000 {}".format(input_file, audio_file))
'''

def process_audio(file_path):
    # print(f"===={file_path}=======")
    t0 = time.time()
    diarization_result = pipeline(file_path)
    # asr_result, info = model.transcribe(file_path, beam_size=1, word_timestamps=False,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    asr_result, info = model.transcribe(file_path, initial_prompt="这是一段会议记录", beam_size=1, word_timestamps=True, \
                                        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    # alignment_model, metadata = whisperx.load_align_model(language_code=info.language, device="cuda")
    # result_aligned = whisperx.align(asr_result, alignment_model, metadata, file_path, "cuda")
    # word_ts = result_aligned["segments"]
    t1 = time.time()
    final_result = diarize_text(asr_result, diarization_result, is_merge = is_merge)
    t2 = time.time()
    output_file = os.path.join(output_dir, os.path.basename(file_path)[:-4] + '.txt')
    with open(output_file, 'w') as f:
        for seg, spk, sent in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}\n'
            f.write(line)
    process_time_audio2txt = t1-t0
    process_time_voiceReco = t2-t1
    print(f'The total duration of {file_path} is {seg.end:.2f}s, and it speed {process_time_audio2txt:.2f}/{process_time_voiceReco:.2f}s for audio2txt/voiceReco respectively, average process {seg.end/process_time_audio2txt:.2f}s/{seg.end/process_time_voiceReco:.2f}s audio per-second.')

# 获取当前目录下所有wav文件名
wav_files = [os.path.join(wave_dir, file) for file in os.listdir(wave_dir)]

# 处理每个wav文件
# with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#     executor.map(process_audio, wav_files)
for wav_file in wav_files:
    process_audio(wav_file)
print('处理完成！')
