import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote_whisper_fold.pyannote_whisper.utils import diarize_text
import torch
import time
import argparse
# import whisperx

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--is_merge', type=str2bool, default='False')
    parser.add_argument('--is_download_wav', type=str2bool, default='True')
    parser.add_argument('--is_resample_by_ffmpeg', type=str2bool, default='False')
    parser.add_argument('--ffmpeg_hz', type=int)
    parser.add_argument('--yt_url_list', nargs='+')
    parser.add_argument('--w2t_type', type=str)
    parser.add_argument('--compute_type', type=str)
    args = parser.parse_args()
    return args

def process_audio(file_path, pipeline_voice, transcriber, w2t_type, is_merge, output_dir):
    # print(f"===={file_path}=======")
    t0 = time.time()
    diarization_result = pipeline_voice(file_path)
    if w2t_type in ["guillaumekln/faster-whisper-large-v2", "Systran/faster-whisper-large-v2"]:
        # asr_result, info = model.transcribe(file_path, beam_size=1, word_timestamps=False,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        # asr_result, info = transcriber(file_path, initial_prompt="这是一段会议记录", beam_size=1, word_timestamps=True, \
        #                                     vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        asr_result, info = transcriber(file_path, beam_size=1, language="zh", \
                                            vad_filter=False, vad_parameters=dict(min_silence_duration_ms=500))
    elif w2t_type in ["BELLE-2/Belle-whisper-large-v2-zh", "BELLE-2/Belle-distilwhisper-large-v2-zh"]:
        # asr_result = transcriber(file_path, vad_filter=True)
        asr_result, info = transcriber(file_path, beam_size=1, language="zh",
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

def main():
    args = parse_args()
    # wave_dir = '/data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/demo/'
    # output_dir = '/data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/out_results/'
    # is_merge = False
    # is_download_wav = True
    # is_resample_by_ffmpeg = False
    # ffmpeg_hz = 6000
    # yt_url_list = [
    #     # 'https://www.youtube.com/watch?v=ChbJfLSI5AE',
    #     'https://www.youtube.com/watch?v=dn8Bs1eXQ8o',
    #     'https://www.youtube.com/watch?v=Hi0Fp_nZSZ0',
    #     'https://www.youtube.com/watch?v=jTo0Kdvz-0E',
    #     # 'https://www.youtube.com/watch?v=tgdJkAx3fJM',
    #     ]
    wave_dir = args.wave_dir
    output_dir = args.output_dir
    is_merge = args.is_merge
    is_download_wav = args.is_download_wav
    is_resample_by_ffmpeg = args.is_resample_by_ffmpeg
    ffmpeg_hz = args.ffmpeg_hz
    yt_url_list = args.yt_url_list
    w2t_type = args.w2t_type
    compute_type = args.compute_type
    if not os.path.exists(wave_dir): os.makedirs(wave_dir)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print("加载声纹模型...")
    pipeline_voice = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1")
    pipeline_voice.to(torch.device("cuda"))

    if w2t_type in ["guillaumekln/faster-whisper-large-v2", "Systran/faster-whisper-large-v2"]:
        print("加载{}语音模型...".format(w2t_type))
        model = WhisperModel(w2t_type, device="cuda", compute_type=compute_type)
        transcriber = model.transcribe
    elif w2t_type in ["BELLE-2/Belle-whisper-large-v2-zh", "BELLE-2/Belle-distilwhisper-large-v2-zh"]:
        print("加载{}语音模型...".format(w2t_type))
        if w2t_type == "BELLE-2/Belle-whisper-large-v2-zh":
            model_path = "./models/BELLE-2/Belle-whisper-large-v2-zh-ct2_model"
        elif w2t_type == "BELLE-2/Belle-distilwhisper-large-v2-zh":
            model_path = "./models/BELLE-2/Belle-distilwhisper-large-v2-zh-ct2_model"
        model = WhisperModel(model_path, device="cuda", compute_type=compute_type, num_workers=1,
                            local_files_only=True)
        transcriber = model.transcribe

        # # 预热
        for _ in range(20):
            _, _ = transcriber("test_16000/demo/Hi0Fp_nZSZ0.wav", beam_size=5)
        # # 语音识别
        # segments, info = model.transcribe(args.audio_path, beam_size=args.beam_size, language=args.language,
        #                                 vad_filter=args.vad_filter)

        # # # # # from transformers import pipeline
        # # # # # transcriber = pipeline(
        # # # # #     "automatic-speech-recognition", 
        # # # # #     model=w2t_type
        # # # # # )
        # # # # # transcriber.model.config.forced_decoder_ids = (
        # # # # #     transcriber.tokenizer.get_decoder_prompt_ids(
        # # # # #         language="zh", 
        # # # # #         task="transcribe"
        # # # # #     )
        # # # # # )

    if is_download_wav:
        # 01 download video
        import yt_dlp
        import demucs.separate
        import shlex
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
        }
        for yt_url in yt_url_list:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download(yt_url)
                video_info = ydl.extract_info(yt_url, download=False)
                file_name = f"{video_info['id']}.wav"
            print("===========")
            print(file_name)
            print("===========")
            
            # 02 preprocess
            if is_resample_by_ffmpeg:
                # file_extension = ['.mp4', '.wav']
                # allowed_files = [file for file in os.listdir() if any(file.lower().endswith(ext) for ext in file_extension)]
                # input_file = max(allowed_files, key=lambda file: os.path.getctime(file))
                input_file = file_name
                demucs.separate.main(shlex.split(f'-n htdemucs --two-stems=vocals "{input_file}" -o "temp_outputs"'))
                input_file = os.path.join(
                        "temp_outputs", "htdemucs", os.path.basename(input_file[:-4]), "vocals.wav")
                audio_file = file_name.replace('.wav', '-{}.wav'.format(ffmpeg_hz)).replace('.mp4', '-{}.wav'.format(ffmpeg_hz)) #"audio_16k.wav"
                os.system("rm -rf {}/{}".format(wave_dir, audio_file))
                print("rm -rf {}/{}".format(wave_dir, audio_file))
                os.system("ffmpeg -i {} -ac 1 -ar {} {}/{}".format(input_file, ffmpeg_hz, wave_dir, audio_file))
                print("ffmpeg -i {} -ac 1 -ar {} {}/{}".format(input_file, ffmpeg_hz, wave_dir, audio_file))
            else:
                os.system("mv ./{} {}/{}".format(file_name, wave_dir, file_name))
        os.system("rm -rf ./temp_outputs")
        print("rm -rf ./temp_outputs")
        os.system("rm -rf ./*.wav")
        print("rm -rf ./*.wav")

    # 获取当前目录下所有wav文件名
    wav_files = [os.path.join(wave_dir, file) for file in os.listdir(wave_dir)]

    # 处理每个wav文件
    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     executor.map(process_audio, wav_files)
    for wav_file in wav_files:
        process_audio(wav_file, pipeline_voice, transcriber, w2t_type, is_merge, output_dir)
    print('处理完成！')

if __name__ == '__main__':
    main()

'''
pc python demo_whisper_pyannote_beta.py \
    --wave_dir /data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/demo/ \
    --output_dir /data01/home/shuchangyong/projects/big_model/whisper-pyannote/test/out_results/ \
    --is_merge False \
    --is_download_wav True \
    --is_resample_by_ffmpeg False \
    --ffmpeg_hz 6000 \
    --w2t_type BELLE-2/Belle-whisper-large-v2-zh \
    --yt_url_list \
    https://www.youtube.com/watch?v=dn8Bs1eXQ8o \
    https://www.youtube.com/watch?v=Hi0Fp_nZSZ0 \
    https://www.youtube.com/watch?v=jTo0Kdvz-0E

'''