from faster_whisper import WhisperModel
import time

model = WhisperModel("guillaumekln/faster-whisper-large-v2", device="cuda", compute_type="float16", local_files_only=False)

t1 = time.time()
segments, info = model.transcribe("audio3.wav", initial_prompt="这是一段会议记录。", beam_size=1, word_timestamps=False,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
t2 = time.time()
print('推理时间')
print(t2-t1)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
