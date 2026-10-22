
import time

import torch
import transformers, accelerate, bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



## 02_04_00

# 0. GPU 세팅 및 점검

# 모델을 올릴 GPU (0번 GPU)
device = "cuda:0"

# - torch.version.cuda: PyTorch가 빌드된 CUDA 버전 (None이면 CPU 전용 빌드)
# - get_device_capability: GPU 아키텍처 (RTX 50 시리즈는 (12, 0) = sm_120)
print("torch:", torch.__version__, "| CUDA 빌드:", torch.version.cuda)
print("transformers:", transformers.__version__, "| accelerate:", accelerate.__version__,
      "| bitsandbytes:", bitsandbytes.__version__)
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다. CUDA용 PyTorch가 설치됐는지 확인하세요."
print("GPU:", torch.cuda.get_device_name(device), "| Compute capability:", torch.cuda.get_device_capability(device))
free, total = torch.cuda.mem_get_info(device)
print(f"VRAM 여유/전체: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
print("='-'" * 30)

# torch: 2.11.0+cu128 | CUDA 빌드: 12.8
# transformers: 4.55.0 | accelerate: 1.8.1 | bitsandbytes: 0.46.0
# GPU: NVIDIA GeForce RTX 5060 Laptop GPU | Compute capability: (12, 0)
# VRAM 여유/전체: 6.9 / 8.0 GB



## 02_04_01

# 모델 ID 정의
# Qwen에서 공개한 대화형 모델입니다.
# Hugging Face에 등록되어 있어 모델 이름으로 바로 불러올 수 있습니다.
model_id = "Qwen/Qwen3-4B-Instruct-2507"

# 1. 4비트 양자화 설정

# BitsAndBytesConfig를 사용해 최신 방식으로 4bit 로드 설정을 지정합니다.
# - load_in_4bit=True: 4비트 양자화 사용
# - bnb_4bit_use_double_quant=True: 2중 양자화를 통해 메모리 사용량 절감
# - bnb_4bit_quant_type="nf4": 최신 NF4 양자화 방식 사용
# - bnb_4bit_compute_dtype=torch.float16: 연산은 FP16으로 수행

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. 토크나이저 로드

# 모델에 맞는 토크나이저를 자동으로 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. 모델 로드

# device_map=device: 02_04_00에서 정한 GPU(cuda:0)에 모델 전체를 올림
# quantization_config=bnb_config: 위에서 정의한 4bit 양자화 설정 적용
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    device_map=device,
    quantization_config=bnb_config
)

print("✅ 모델 로드 완료!")

# 모델 메모리 사용량 출력 (GB 단위)
print(f"모델의 메모리 사용량: {model.get_memory_footprint() / 1024**3:.2f} GB")

# 모델의 메모리 사용량: 2.42 GB

# 4. GPU 로딩 확인

# 모든 파라미터가 GPU(cuda:0)에 올라갔는지 확인합니다.
param_devices = {str(p.device) for p in model.parameters()}
print("파라미터 위치:", param_devices)
assert param_devices == {device}, f"GPU 밖에 있는 파라미터가 있습니다: {param_devices}"

# 4비트 양자화가 실제로 적용됐는지 확인합니다. 선형층이 Linear4bit로 바뀌어 있어야 합니다.
n_4bit = sum(isinstance(m, bitsandbytes.nn.Linear4bit) for m in model.modules())
print("4비트 선형층 개수:", n_4bit)
assert n_4bit > 0, "4비트 양자화가 적용되지 않았습니다."
print("예시 (0번 층 q_proj):", model.model.layers[0].self_attn.q_proj)
print(f"GPU 할당 메모리: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print("='-'" * 30)

# 파라미터 위치: {'cuda:0'}
# 4비트 선형층 개수: 252          <- 36개 층 × (q, k, v, o, gate, up, down) 7개
# 예시 (0번 층 q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)

# GPU 할당 메모리: 2.49 GB


## 02_04_02


chat_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "LLM에 대해 5문장으로 요약해줘."}
]
# 2. 챗 템플릿 적용 및 토큰화

input_ids = tokenizer.apply_chat_template(
    chat_history,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 입력 텐서도 모델과 같은 GPU에 있어야 합니다.
print("입력 텐서 위치:", input_ids.device)

# 3. 텍스트 생성 (추론)

torch.cuda.reset_peak_memory_stats(device)
start = time.perf_counter()
output = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
# GPU 연산은 비동기로 진행되므로, 끝날 때까지 기다린 뒤 시간을 잽니다.
torch.cuda.synchronize(device)
elapsed = time.perf_counter() - start

# 생성 속도와 추론 중 최대 GPU 메모리를 확인합니다.
new_tokens = output.shape[-1] - input_ids.shape[-1]
print(f"생성 토큰: {new_tokens}개, {elapsed:.1f}초 ({new_tokens / elapsed:.1f} tokens/s)")
print(f"추론 중 최대 GPU 메모리: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

# 입력 텐서 위치: cuda:0
# 생성 토큰: 180개, 11.9초 (15.1 tokens/s)
# 추론 중 최대 GPU 메모리: 2.55 GB

# 4. 생성된 결과를 텍스트로 변환 및 출력
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("생성된 답변:\n")
print(generated_text)

# 생성된 답변:

# system
# You are a helpful assistant.
# user
# LLM에 대해 5문장으로 요약해줘.
# assistant
# LLM은 대규모 언어 모델로, 수십억 개의 파라미터를 가진 자연어 처리 기반의 인공지능 모델이다.
# 주요 목적은 자연어를 이해하고 생성하는 데 있어 인간과 유사한 성능을 달성하는 것이다.
# LLM은 텍스트 데이터를 통해 학습되며, 질문에 답하거나 글을 생성하는 등 다양한 작업에 활용된다.
# 그러나 텍스트 생성에 있어 오류나 편향, 그리고 정보의 정확성에 대한 의심이 존재한다.
# 최근에는 기술적 발전으로 인해 다양한 분야에서 실제 응용이 가능해지고 있다.