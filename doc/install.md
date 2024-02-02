## Environment Setup
step 1. Install environment for pytorch training
```
conda create --name whisper_pyannote python=3.8.5
conda activate whisper_pyannote
pip install yt_dlp
```

<!-- conda install ffmpeg #在自己虚拟环境下用conda安装 # sudo apt update && sudo apt install ffmpeg

pip install setuptools-rust

pip install git+https://github.com/facebookresearch/demucs#egg=demucs
pip install openai tiktoken
pip3 install torch torchvision torchaudio
pip install git+https://github.com/m-bain/whisperx.git
pip install faster_whisper
apt install -y ffmpeg sox libsndfile1
pip install --upgrade hydra-core llvmlite omegaconf --ignore-installed
python -m pip install git+https://github.com/NVIDIA/NeMo.git@main
pip install --upgrade Cython jiwer braceexpand webdataset librosa sentencepiece
pip install --upgrade youtokentome pyannote-audio transformers pandas inflect editdistance
pip install -U pytorch-lightning

pip install git+https://github.com/NVIDIA/NeMo.git
pip install hydra-core
pip install datasets
pip install lhotse

pip install pyannote.audio -->




step 2. 语音&声纹结果合并包环境安装

```
git clone https://github.com/yinruiqing/pyannote-whisper
mv pyannote-whisper pyannote_whisper_fold
cd pyannote_whisper_fold
git checkout c66ee03b
sudo apt-get install libsndfile1-dev
pip install -r requirements.txt
pip install -v -e .
```

huggingface token码
```
huggingface-cli login
```

pyannote_whisper_fold/pyannote_whisper/utils.py中代码修改

 - (1) get_text_with_timestamp(transcribe_res)函数定义修改为
```
def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    # for item in transcribe_res['segments']:
    #     start = item['start']
    #     end = item['end']
    #     text = item['text']
    #     timestamp_texts.append((Segment(start, end), text))
    for item in transcribe_res:
        start = item.start
        end = item.end
        text = item.text
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts
```

 - (2) diarize_text(transcribe_res, diarization_result, is_merge=True)函数定义修改为
```
def diarize_text(transcribe_res, diarization_result, is_merge=True):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    if is_merge == True:
        res_processed = merge_sentence(spk_text)
        return res_processed
    else:
        return spk_text
```


step 3. faster_whisper 语音模型
```
pip install faster-whisper
```

step 4. whisperx
```
pip install git+https://github.com/m-bain/whisperx.git
```

step 5. 安装ffmpeg
(1) 在自己虚拟环境下用conda安装
```
conda install ffmpeg  
```
(2) 系统环境下
```
sudo apt update && sudo apt install ffmpeg
```

step 6. others
```
pip install git+https://github.com/facebookresearch/demucs#egg=demucs
pip install --upgrade youtokentome pyannote-audio transformers pandas inflect editdistance
pip install --upgrade transformers
```
