#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-Math-1.5B-Instruct')
print(model_dir)  # 输出模型存储路径