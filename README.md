
1. [Env install](doc/install.md)
2. Quick test
  - wave_dir: 带下载音频存储路径
  - output_dir: 音频声纹处理结果保存路径
  - is_merge: 处理结果是否将语句块合并为一个无标点长句
  - is_download_wav: 是否重新下载音频(如果已经下载好了，可以不用反复下载)
  - is_resample_by_ffmpeg: 是否重新对音频文件进行采样到指定的频率(ffmpeg_hz)
  - ffmpeg_hz: 设定重新采样的指定频率
  - yt_url_list: 带下载的音频或视频url
```
python demo_whisper_pyannote_beta.py \
    --wave_dir ./test/demo/ \
    --output_dir ./test/out_results/ \
    --is_merge False \
    --is_download_wav True \
    --is_resample_by_ffmpeg False \
    --ffmpeg_hz 6000 \
    --yt_url_list \
    https://www.youtube.com/watch?v=dn8Bs1eXQ8o \
    https://www.youtube.com/watch?v=Hi0Fp_nZSZ0 \
    https://www.youtube.com/watch?v=jTo0Kdvz-0E

```
