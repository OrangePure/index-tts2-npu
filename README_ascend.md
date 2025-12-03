### 准备环境
容器启动，以 CANN 8.3.rc1，910b 机器，ubuntu22.04 系统，py3.11 为例：
```
container_name=indextts-ascend
workspace_path=/workspace
docker run \
    --name ${container_name} \
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
    -v ${workspace_path}:${workspace_path} \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11  bash
```
进入容器
```
docker exec -it ${container_name} /bin/bash
```
项目下载

```
git clone https://github.com/triomino/index-tts.git && cd index-tts
git lfs pull # 如果需要项目中的样例音频
```
### 安装依赖
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
安装原仓库依赖
```
pip install -e .
```
最后安装 torch_npu，`torch_npu(v2.8.0-7.2.0)` 需要单独安装。
其中 torch_npu 需要从[源码](https://gitcode.com/Ascend/pytorch)编译，切到分支 v2.8.0-7.2.0. 编译过程中需要升级 gcc 和 cmake，见[torch_npu源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)


模型拉取，连 hf 太慢可以改成走 modelscope，或者切 hf 镜像源

```
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
export HF_ENDPOINT="https://hf-mirror.com"
```
### 性能测试&样例
```
# benchmark
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=1
# export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
ASCEND_RT_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PYTHONPATH="$PYTHONPATH:."  python indextts/benchmark.py
```
样例
```python
import os
from indextts.infer_v2_npu import IndexTTS2NPU

# 1. Initialize
tts = IndexTTS2NPU(
    cfg_path="checkpoints/config.yaml", 
    model_dir="checkpoints", 
    use_cuda_kernel=True, 
    use_fp16=True, 
    static=True
)

# 2. Prepare inputs
prompt_wav = "examples/voice_01.wav"
text = "IndexTTS2 is amazing on Ascend NPU! It supports fast static graph inference."

# 3. Run inference
tts.infer_fast(
    spk_audio_prompt=prompt_wav, 
    text=text, 
    output_path="output_npu.wav", 
    verbose=True
)
```


