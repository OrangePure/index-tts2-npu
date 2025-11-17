容器启动
```
docker run \
    --name zya-tts-scratch \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --net=host \
    --shm-size=256g \
    --privileged=true \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/zya/workspace/scratch:/home/zya/workspace/scratch \
    -itd quay.io/ascend/vllm-ascend:v0.10.2rc1  bash
```

项目下载

```
git clone https://github.com/triomino/index-tts.git && cd index-tts
git lfs pull # 如果需要项目中的样例音频
```

WeTextProcessing pip 直接安装可能会出错，单独进行安装：

```
# 下载安装包并解压
wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz
tar -zxvf openfst-1.8.3.tar.gz
# 进入目录后编译安装
cd openfst-1.8.3
./configure --enable-far --enable-mpdt --enable-pdt
make -j$(nproc)
make install
# 确认动态库文件存在：
ls /usr/local/lib/libfstmpdtscript.so.26
# 配置动态库路径
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig
# 安装WeTextProcessing
pip3 install WeTextProcessing==1.0.4.1
```

```
pip install -e .
# torch, torch-npu, torchaudio 得重新安装下，安装昇腾匹配的版本
pip install torch==2.7.1+cpu torch-npu==2.7.1.dev20250724 torchaudio==2.7.1
```

模型拉取，连 hf 太慢可以改成走 modelscope，或者切 hf 镜像源

```
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
export HF_ENDPOINT="https://hf-mirror.com"
```

```
# 昇腾推理样例
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=1
# export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
ASCEND_RT_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PYTHONPATH="$PYTHONPATH:."  python indextts/infer_v2_npu.py
```




