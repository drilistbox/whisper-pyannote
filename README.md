
1. [Env install](doc/install.md)
2. Quick test
  - wave_dir: 带下载音频存储路径
  - output_dir: 音频声纹处理结果保存路径
  - is_merge: 处理结果是否将语句块合并为一个无标点长句
  - is_download_wav: 是否重新下载音频(如果已经下载好了，可以不用反复下载)
  - is_resample_by_ffmpeg: 是否重新对音频文件进行采样到指定的频率(ffmpeg_hz)
  - ffmpeg_hz: 设定重新采样的指定频率
  - w2t_type: 音频转文本模型类型
  ```
      --w2t_type Systran/faster-whisper-large-v2
      --w2t_type guillaumekln/faster-whisper-large-v2
      --w2t_type BELLE-2/Belle-whisper-large-v2-zh
      --w2t_type BELLE-2/Belle-distilwhisper-large-v2-zh
  ```
  - compute_type: 计算精度类型
  ```
  float16
  int8_float16
  int8
  ```
  - yt_url_list: 带下载的音频或视频url
```
conda activate whisper_pyannote
pc python demo_whisper_pyannote_beta.py \
    --wave_dir ./test_16000/demo/ \
    --output_dir ./test_16000/out_results/ \
    --is_merge False \
    --is_download_wav False \
    --is_resample_by_ffmpeg False \
    --w2t_type Systran/faster-whisper-large-v2 \
    --ffmpeg_hz 16000 \
    --compute_type float16 \
    --yt_url_list \
    https://www.youtube.com/watch?v=Hi0Fp_nZSZ0 \
    https://www.youtube.com/watch?v=dn8Bs1eXQ8o \
    https://www.youtube.com/watch?v=jTo0Kdvz-0E

```
