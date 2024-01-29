1. 环境搭建
    1.1 pyannote 声纹模型
    pip install pyannote.audio
    pip install -r requirements.txt

    1.2 faster_whisper 语音模型
    我用的是faster-whisper-large-v2版本,对应的路径是当前工程目录下：
    guillaumekln/faster-whisper-large-v2 
    如果有已经下好的faster-whisper语音模型可直接链接至当前工程目录下
    在调用语音模型时修改成自己的目录即可；
    如果还没下语音模型，可在这模型主页下载：
    https://huggingface.co/guillaumekln/faster-whisper-large-v2/tree/main
    或直接运行以下命令进行下载：
    git clone git@hf.co:guillaumekln/faster-whisper-large-v2
    检查语音识别目录下requirements.txt中所需包是否安装

    1.3 语音&声纹结果合并包环境安装
    cd pyannote_whisper_fold
    pip install -r requirements.txt

2. 推理代码
    语音识别推理： demo_whisper.py
    声纹识别推理： demo_pyannote.py
    语音+声纹结果合并推理： demo_whisper_pyannote.py    


3. 声纹评测代码
    声纹评测代码需下载数据集，我下了AMI数据集放到database中
    调用cacul_matrix.py即可，需要在代码开头环境变量中添加数据yml路径
    import os
    os.environ["PYANNOTE_DATABASE_CONFIG"] = "database/database.yml"
    声纹数据集数据格式可参考文档
    https://houmo.feishu.cn/docx/FMn9dsKmvoyjibxpSagcPdAgngb