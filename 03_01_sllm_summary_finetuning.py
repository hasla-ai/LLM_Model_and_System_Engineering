#3장 sLLM
# Chapter 3 Supervised Fine-Tuning (SFT) 및 학습 효율화

# 03_01_1

import os
import torch
import transformers, accelerate, trl, peft, datasets


## 03_01_00

# 0. GPU 세팅 및 점검

device = "cuda:0"

print("torch:", torch.__version__, "| CUDA 빌드:", torch.version.cuda)
print("transformers:", transformers.__version__, "| accelerate:", accelerate.__version__,
      "| trl:", trl.__version__, "| peft:", peft.__version__, "| datasets:", datasets.__version__)
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다. CUDA용 PyTorch가 설치됐는지 확인하세요."
print("GPU:", torch.cuda.get_device_name(device), "| Compute capability:", torch.cuda.get_device_capability(device))
# 학습 설정(SFTConfig)이 bf16=True를 쓰므로 GPU가 bf16을 지원해야 합니다.
assert torch.cuda.is_bf16_supported(), "이 GPU는 bf16을 지원하지 않습니다."
free, total = torch.cuda.mem_get_info(device)
print(f"VRAM 여유/전체: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
print("='-'" * 30)


## 03_01_01

# RunPod에서 작은 컨테이너 디스크 대신 큰 볼륨(/workspace)에 모델을 받으려는 경우

# cache_dir = "/workspace/huggingface_cache"
# os.environ['HF_HOME'] = cache_dir
# os.environ['TRANSFORMERS_CACHE'] = cache_dir
# os.environ['DATASETS_CACHE'] = cache_dir
# os.makedirs(cache_dir, exist_ok=True)
#
# print(f" Hugging Face 캐시 디렉토리가 대용량 스토리지로 설정되었습니다:")
# print(f"{os.environ.get('HF_HOME')}")