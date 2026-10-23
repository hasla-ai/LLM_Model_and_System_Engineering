'''
상품 이미지(60×80)와 상품명을 보고, 7가지 패션 속성을 JSON으로 답하는 VLM 모델.

7가지 속성은
1) gender
2) masterCategory
3) subCategory
4) season
5) usage
6) baseColour
7) articleType

(성별, 대분류, 중분류, 시즌, 용도, 색상, 품목)입니다.

데이터셋은 `ashraq/fashion-product-images-small`입니다.
원본 44,072건. 상품명(productDisplayName), 속성 열들, image(PIL 60×80).

베이스 모델은 Qwen3-VL-4B(이미지와 텍스트를 함께 처리).
4비트로 압축한 뒤 LoRA로 학습.

주요 함수는 07 collate_fn.
 - 글은 토큰으로, 이미지는 픽셀 패치로 변환합니다.

'''

import torch
import transformers
import accelerate
import datasets
import trl
import peft
import qwen_vl_utils

import io
import json
import os
import random

import numpy as np
import torch
import wandb

from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig


## 04_01_00

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

torch.cuda.set_per_process_memory_fraction(0.85, device)

# 어텐션 커널: Linux(WSL)는 FlashAttention, FlashAttention이 없는 Windows는 GQA 경로를 끄고 memory-efficient 커널
# 학습 중 L×L 행렬을 만드는 math 방식으로 넘어가지 않도록 커널을 고정해, 못 쓰면 바로 오류가 나게 한다.
from torch.nn.attention import sdpa_kernel, SDPBackend

flash_ok = torch.backends.cuda.is_flash_attention_available()
if not flash_ok:
    import transformers.integrations.sdpa_attention as sdpa_attention
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
attn_backend = SDPBackend.FLASH_ATTENTION if flash_ok else SDPBackend.EFFICIENT_ATTENTION
print("FlashAttention 빌드:", flash_ok, "| 학습 어텐션 커널:", attn_backend.name)


print(f"VRAM 여유/전체: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
print("='-'" * 30)

# torch: 2.8.0+cu128 | CUDA 빌드: 12.8
# transformers: 4.57.6 | accelerate: 1.14.0 | trl: 0.29.1 | peft: 0.19.1 | datasets: 3.6.0
# GPU: NVIDIA GeForce RTX 5060 Laptop GPU | Compute capability: (12, 0)
# FlashAttention 빌드: True | 학습 어텐션 커널: FLASH_ATTENTION
# VRAM 여유/전체: 6.9 / 8.0 GB
# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'


# 04_01_01 기본설정

wandb.init(mode="disabled")

SEED = 42
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = "output_dir_qwen3_vl_4b_instruct_lora"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device name:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("gpu:", torch.cuda.get_device_name(0))


# torch version: 2.8.0+cu128
# cuda available: True
# cuda version: 12.8
# device name: NVIDIA GeForce RTX 5060 Laptop GPU
# capability: (12, 0)
# gpu: NVIDIA GeForce RTX 5060 Laptop GPU

print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)
print("trl:", trl.__version__)
print("peft:", peft.__version__)

# transformers: 4.57.6
# accelerate: 1.14.0
# datasets: 3.6.0
# trl: 0.29.1
# peft: 0.19.1




# 04_01_02 프롬프트 정의

# 모델에게 줄 지시문
# system_message	모델의 역할 ("패션/스타일 정보를 추론하는 분류 모델")
# prompt: 사용자 지시문 틀. {name} 자리에 상품명이 들어가고, 7개 키·JSON 형식·예시를 알려 줍니다.
# {{ }}는 .format()이 중괄호를 건드리지 않게 막는 표기입니다.

system_message = "당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다."

prompt = """입력 정보:
- name: {name}
- image: [image]

위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:
1) gender
2) masterCategory
3) subCategory
4) season
5) usage
6) baseColour
7) articleType

출력 시 **아래 JSON 예시 형태**를 반드시 지키세요:
{{
  "gender": "예시값",
  "masterCategory": "예시값",
  "subCategory": "예시값",
  "season": "예시값",
  "usage": "예시값",
  "baseColour": "예시값",
  "articleType": "예시값"
}}

# 예시
{{
  "gender": "Men",
  "masterCategory": "Accessories",
  "subCategory": "Eyewear",
  "season": "Winter",
  "usage": "Casual",
  "baseColour": "Blue",
  "articleType": "Sunglasses"
}}

# 주의
- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.
"""

# 04_01_03 데이터셋 로드 및 라벨 설정

## 표 데이터 → 대화 형식
## dataset
## 원본 44,072건. 상품명(productDisplayName), 속성 열들, image(PIL 60×80)

def combine_cols_to_label(example):
    '''
    7개 속성 열을 JSON 문자열 하나로 묶어 label(정답)을 만듭니다.
    '''

    label_dict = {
        "gender": example["gender"],
        "masterCategory": example["masterCategory"],
        "subCategory": example["subCategory"],
        "season": example["season"],
        "usage": example["usage"],
        "baseColour": example["baseColour"],
        "articleType": example["articleType"],
    }

    example["label"] = json.dumps(label_dict, ensure_ascii=False)
    return example

def format_data(sample):
    '''
    한 행을 messages로 바꿈.
    system(역할) / user(지시문 + 이미지) / assistant(정답 JSON).
    '''

    buffer = io.BytesIO()
    sample["image"].save(buffer, format="PNG")
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")

    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(name=sample["productDisplayName"]),
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["label"],
                    }
                ],
            },
        ],
    }

dataset = load_dataset("ashraq/fashion-product-images-small", split="train")

# dataset_add_label
## label을 추가하고 seed=4242로 섞음. 먼저 섞어서 순서 쏠림이 없음.

dataset_add_label = dataset.map(combine_cols_to_label)
dataset_add_label = dataset_add_label.shuffle(seed=4242)

# format_data()
# 한 행을 messages로 바꿈
# system(역할) / user(지시문 + 이미지) / assistant(정답 JSON)

# formatted_dataset	44,072건의 messages 목록 (Python 리스트)

formatted_dataset = [format_data(row) for row in dataset_add_label]

# test_size=0.9. 학습 4,407건(10%), 테스트 39,665건(90%)

train_dataset, test_dataset = train_test_split(
    formatted_dataset,
    test_size=0.9,
    random_state=42,
)

print("학습 데이터의 개수:", len(train_dataset))
print("테스트 데이터의 개수:", len(test_dataset))
print("샘플 데이터:")
print(train_dataset[0])

# 학습 데이터의 개수: 4407
# 테스트 데이터의 개수: 39665
# 샘플 데이터:
# {'messages': [{'role': 'system', 'content': [{'type': 'text', 'text': '당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '입력 정보:\n- name: Mr.Men Men\'s Charcoal White T-shirt\n- image: [image]\n\n위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:\n1) gender\n2) masterCategory\n3) subCategory\n4) season\n5) usage\n6) baseColour\n7) articleType\n\n출력 시 **아래 JSON 예시 형태**를 반드시 지키세요:\n{\n  "gender": "예시값",\n  "masterCategory": "예시값",\n  "subCategory": "예시값",\n  "season": "예시값",\n  "usage": "예시값",\n  "baseColour": "예시값",\n  "articleType": "예시값"\n}\n\n# 예시\n{\n  "gender": "Men",\n  "masterCategory": "Accessories",\n  "subCategory": "Eyewear",\n  "season": "Winter",\n  "usage": "Casual",\n  "baseColour": "Blue",\n  "articleType": "Sunglasses"\n}\n\n# 주의\n- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.\n'}, {'type': 'image', 'image': <PIL.Image.Image image mode=RGB size=60x80 at 0x7A72E5E70CE0>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "season": "Fall", "usage": "Casual", "baseColour": "Grey", "articleType": "Tshirts"}'}]}]}



# 04_01_04 프로세서 및 모델 로드

# 책 설정은 bf16 약 9GB, LoRA r=64를 모든 선형층에, 배치 2·누적 8

# 4비트 양자화(BitsAndBytesConfig)로 모델 로드 = QLoRA의 Q 부분 (LoRA는 04_01_10의 SFTTrainer에서 한 번 붙임)
# 4비트로 불러오면 약 3GB가 됩니다(LLM 4비트 약 1.9GB + 임베딩 bf16 0.8GB + 비전 인코더).

# 배치 1·누적 16 (유효 배치 16은 그대로)
# 이미지 해상도(max_pixels) 조정
# VRAM 상한과 사전 시험(TEST_RUN)은 03_01과 같은 방식으로 적용

from transformers import BitsAndBytesConfig

# 28*28은 Qwen2.5-VL의 값입니다(패치 14 × 2×2 병합).
# Qwen3-VL은 패치가 16이라, 시각 토큰 하나가 32×32 픽셀입니다.
# 데이터 이미지는 60×80 픽셀(03 출력에서 확인)인데,
# min_pixels=256*28*28(약 20만 픽셀)이라 이미지를 6배 넘게 키워서 약 196개 토큰을 만듭니다.
# 원본에 없는 정보를 늘려 쓰는 셈이라 메모리만 듭니다.
# 시각 토큰을 196개에서 64개로 줄이면 샘플 길이와 시간이 줄어듭니다.
# 정확도와 어떻게 맞바꾸는지 비교해 보면 좋은 연습이 됩니다.


# processor

# 토크나이저와 이미지 처리기를 묶은 processor를 설정합니다.
# min_pixels/max_pixels로 이미지 크기를 정하고, 그에 따라 시각 토큰 수가 정해집니다(약 64개).
# 참고) padding_side="right"는 학습용 설정.

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    # min_pixels=256 * 28 * 28,
    # max_pixels=512 * 28 * 28,
    min_pixels=64 * 32 * 32,     # 60×80 이미지 → 약 64토큰 (Qwen3-VL 기본 최소값과 같음)
    max_pixels=256 * 32 * 32,
)

processor.tokenizer.padding_side = "right"

# 4비트 압축설정

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # RTX 5060은 bf16 지원, 학습도 bf16
)

# 4비트로 압축된 Qwen3-VL. 비전 인코더와 텍스트 디코더로 구성.
# 학습 중에 생성용 캐시를 쓰지 않음(`use_cache=False`).

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    # device_map="auto",
    # torch_dtype=torch.bfloat16,
    device_map=device,
    dtype=torch.bfloat16,                    # transformers 4.57에서 torch_dtype은 사용 중단 예정
    quantization_config=bnb_config,
)

model.config.use_cache = False

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# 로드 확인

print(model.get_memory_footprint() / 1024**3)

# 2.616174042224884


# 04_01_05 채팅 탬플릿 확인    

# text: 첫 샘플에 템플릿을 적용한 문자열은 아래와 같음.
# <|im_start|>system … user … <|vision_start|><|image_pad|><|vision_end|> … assistant {정답}<|im_end|>.
# 이미지는 아직 자리표시 토큰 하나이고, 07에서 실제 시각 토큰 개수만큼 늘어납니다.


text = processor.apply_chat_template(
    train_dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print("채팅 템플릿 적용 결과:")
print(text)

# 채팅 템플릿 적용 결과:
# <|im_start|>system
# 당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.<|im_end|>
# <|im_start|>user
# 입력 정보:
# - name: Mr.Men Men's Charcoal White T-shirt
# - image: [image]

# 위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:
# 1) gender
# 2) masterCategory
# 3) subCategory
# 4) season
# 5) usage
# 6) baseColour
# 7) articleType

# 출력 시 **아래 JSON 예시 형태**를 반드시 지키세요:
# {
#   "gender": "예시값",
#   "masterCategory": "예시값",
#   "subCategory": "예시값",
#   "season": "예시값",
#   "usage": "예시값",
#   "baseColour": "예시값",
#   "articleType": "예시값"
# }

# # 예시
# {
#   "gender": "Men",
#   "masterCategory": "Accessories",
#   "subCategory": "Eyewear",
#   "season": "Winter",
#   "usage": "Casual",
#   "baseColour": "Blue",
#   "articleType": "Sunglasses"
# }

# # 주의
# - 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.
# <|vision_start|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "season": "Fall", "usage": "Casual", "baseColour": "Grey", "articleType": "Tshirts"}<|im_end|>



# 04_01_06 LoRA 설정. 04_01_10 에 붙임.

## peft_config: r=64(A·B 크기), lora_alpha=64(배율 alpha/r = 1), lora_dropout=0.05, 텍스트 디코더의 7종 선형층에만 붙임


peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=64,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

# 04_01_07_Collate_fn_설정
## 07 collate_fn: 샘플 → GPU 입력 묶음


def get_special_token_ids(tokenizer):
    '''
    이미지 관련 특수 토큰(vision_start/end, image_pad 등)의 ID 목록.
    정답에서 뺄 때 쓴다.
    '''
    special_tokens = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]

    token_ids = []

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            continue

        if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
            continue

        token_ids.append(token_id)

    return list(set(token_ids))

def find_subsequence(sequence, subsequence):
    '''
    토큰 목록에서 <|im_start|>assistant\n이 시작하는 위치를 찾는다.
    '''
    if len(subsequence) == 0:
        return -1

    last_match = -1

    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            last_match = i

    return last_match

def mask_non_assistant_tokens(labels, input_ids, tokenizer):
    '''
    답변 앞부분을 모두 -100으로 바꿔 정답 JSON만 학습하도록 한다.
    '''
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = tokenizer.encode(
        assistant_prefix,
        add_special_tokens=False,
    )

    for row_idx in range(input_ids.shape[0]):
        row_input_ids = input_ids[row_idx].tolist()

        start_idx = find_subsequence(
            row_input_ids,
            assistant_prefix_ids,
        )

        if start_idx == -1:
            labels[row_idx, :] = -100
            continue

        answer_start_idx = start_idx + len(assistant_prefix_ids)
        labels[row_idx, :answer_start_idx] = -100

    return labels

def collate_fn(examples):
    '''
    템플릿 적용 → 이미지 꺼내기(process_vision_info) → processor로 변환
    → labels를 만들고 패딩·이미지 토큰·답변 앞부분을 -100으로 처리.
    '''
    texts = [
        processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for example in examples
    ]

    image_inputs = []
    video_inputs = []

    for example in examples:
        image_input, video_input = process_vision_info(example["messages"])

        if image_input is not None:
            image_inputs.extend(image_input)

        if video_input is not None and len(video_input) > 0:
            video_inputs.extend(video_input)

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
    }

    if len(image_inputs) > 0:
        processor_kwargs["images"] = image_inputs

    if len(video_inputs) > 0:
        processor_kwargs["videos"] = video_inputs

    batch = processor(**processor_kwargs)

    labels = batch["input_ids"].clone()

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    for special_token_id in get_special_token_ids(processor.tokenizer):
        labels[labels == special_token_id] = -100

    labels = mask_non_assistant_tokens(
        labels=labels,
        input_ids=batch["input_ids"],
        tokenizer=processor.tokenizer,
    )

    batch["labels"] = labels

    # input_ids(토큰), attention_mask, pixel_values(이미지 패치 값),
    # image_grid_thw(이미지 격자 크기: 시간 × 높이 × 너비 패치 수), labels

    return batch


# 04_01_08_Collate_fn_테스트

# 첫 샘플 하나로 batch를 만들어, 각 텐서의 모양과 실제 학습 대상 토큰 수, 정답 디코딩 결과를 확인함.
# image_grid_thw로 시각 토큰이 몇 개인지 알 수 있습니다(높이 × 너비 패치 수 ÷ 4).


# 단일 예시 확인
example = train_dataset[0]  # 데이터셋의 첫 번째 아이템

print("단일 예시 데이터:")
print(example)

# 단일 예시 데이터:
# {'messages': [{'role': 'system', 'content': [{'type': 'text', 'text': '당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '입력 정보:\n- name: Mr.Men Men\'s Charcoal White T-shirt\n- image: [image]\n\n위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:\n1) gender\n2) masterCategory\n3) subCategory\n4) season\n5) usage\n6) baseColour\n7) articleType\n\n출력 시 **아래 JSON 예시 형태**를 반드시 지키세요:\n{\n  "gender": "예시값",\n  "masterCategory": "예시값",\n  "subCategory": "예시값",\n  "season": "예시값",\n  "usage": "예시값",\n  "baseColour": "예시값",\n  "articleType": "예시값"\n}\n\n# 예시\n{\n  "gender": "Men",\n  "masterCategory": "Accessories",\n  "subCategory": "Eyewear",\n  "season": "Winter",\n  "usage": "Casual",\n  "baseColour": "Blue",\n  "articleType": "Sunglasses"\n}\n\n# 주의\n- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.\n'}, {'type': 'image', 'image': <PIL.Image.Image image mode=RGB size=60x80 at 0x74C04E547A40>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "season": "Fall", "usage": "Casual", "baseColour": "Grey", "articleType": "Tshirts"}'}]}]}

# collate_fn 테스트 (배치 크기 1로)
batch = collate_fn([example])

print("\n처리된 배치 데이터:")
print("입력 ID 형태:", batch["input_ids"].shape)
print("어텐션 마스크 형태:", batch["attention_mask"].shape)

# 처리된 배치 데이터:
# 입력 ID 형태: torch.Size([1, 429])
# 어텐션 마스크 형태: torch.Size([1, 429])

if "pixel_values" in batch:
    print("이미지 픽셀 형태:", batch["pixel_values"].shape)

if "image_grid_thw" in batch:
    print("이미지 grid 형태:", batch["image_grid_thw"].shape)
    print("이미지 grid 값:")
    print(batch["image_grid_thw"])

if "video_grid_thw" in batch:
    print("비디오 grid 형태:", batch["video_grid_thw"].shape)
    print("비디오 grid 값:")
    print(batch["video_grid_thw"])

print("레이블 형태:", batch["labels"].shape)

# 이미지 픽셀 형태: torch.Size([280, 1536])
# 이미지 grid 형태: torch.Size([1, 3])
# 이미지 grid 값:
# tensor([[ 1, 20, 14]])
# 레이블 형태: torch.Size([1, 429])


# 입력 ID 전체 확인
print("\n입력에 대한 정수 인코딩 결과:")
print(batch["input_ids"][0])

# 입력에 대한 정수 인코딩 결과:
# tensor([151644,   8948,    198,  64795,  82528,  33704,  90667,  21329,  80573,
#         138017,  79632,   3153,      8,  42039, 126558,  45104,    101,  92031,
#             14, 141274,  32077,  60039,  18411,  57835, 126605,  42905, 128618,
#          97929,  54070, 142713,  78952,     13, 151645,    198, 151644,    872,
#            198,  43866,  28754,  60039,    510,     12,    829,     25,   4392,
#           1321,    268,  11012,    594,   4864,  40465,   5807,    350,  33668,
#            198,     12,   2168,     25,    508,   1805,   2533,  80901,  60039,
#          18411,  81718, 144059,  42039,     11, 136646,    220,     22,  19969,
#          21329,   1376,  19391, 128605,  93668,   4718, 141966,  17380,  57835,
#         126605,  33883,  55673,  50302,    510,     16,      8,   9825,    198,
#             17,      8,   7341,   6746,    198,     18,      8,   1186,   6746,
#            198,     19,      8,   3200,    198,     20,      8,  10431,    198,
#             21,      8,   2331,  33281,    198,     22,      8,   4549,    929,
#            271,  53496,  57133,  44518,   3070,  52959,  53442,   4718,  95617,
#          29326, 141966,    334,  18411, 141762,  66790, 126896,  50302,    510,
#            515,    220,    330,  12968,    788,    330, 127027,  29326,  82801,
#            756,    220,    330,  13629,   6746,    788,    330, 127027,  29326,
#          82801,    756,    220,    330,   1966,   6746,    788,    330, 127027,
#          29326,  82801,    756,    220,    330,  16798,    788,    330, 127027,
#          29326,  82801,    756,    220,    330,  17698,    788,    330, 127027,
#          29326,  82801,    756,    220,    330,   3152,  33281,    788,    330,
#         127027,  29326,  82801,    756,    220,    330,   7058,    929,    788,
#            330, 127027,  29326,  82801,    698,    630,      2,  95617,  29326,
#            198,    515,    220,    330,  12968,    788,    330,  28719,    756,
#            220,    330,  13629,   6746,    788,    330,   6054,   2433,    756,
#            220,    330,   1966,   6746,    788,    330,  97454,  98128,    756,
#            220,    330,  16798,    788,    330,  66188,    756,    220,    330,
#          17698,    788,    330,  49242,    928,    756,    220,    330,   3152,
#          33281,    788,    330,  10331,    756,    220,    330,   7058,    929,
#            788,    330,     50,   2185,  33868,    698,    630,      2,  55673,
#          20401,    198,     12,    220,     22,  59761, 142654,  87608,  23084,
#         128792,  20401,  60039,      7, 144153,  53189,     11,  53435,  40853,
#          77002,      8,  16560,  18585,    230,  66845, 133970,  87425,  95577,
#          50302,    624, 151652, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#         151655, 151653, 151645,    198, 151644,  77091,    198,   4913,  12968,
#            788,    330,  28719,    497,    330,  13629,   6746,    788,    330,
#           2164,  30431,    497,    330,   1966,   6746,    788,    330,   5366,
#          22744,    497,    330,  16798,    788,    330,  49772,    497,    330,
#          17698,    788,    330,  49242,    928,    497,    330,   3152,  33281,
#            788,    330,  59165,    497,    330,   7058,    929,    788,    330,
#             51,    927,  19666,   9207, 151645,    198])


# 레이블 전체 확인
print("\n레이블에 대한 정수 인코딩 결과:")
print(batch["labels"][0])

# 레이블에 대한 정수 인코딩 결과:
# tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
#           -100,   -100,   -100,   -100,   -100,   -100,   -100,   4913,  12968,
#            788,    330,  28719,    497,    330,  13629,   6746,    788,    330,
#           2164,  30431,    497,    330,   1966,   6746,    788,    330,   5366,
#          22744,    497,    330,  16798,    788,    330,  49772,    497,    330,
#          17698,    788,    330,  49242,    928,    497,    330,   3152,  33281,
#            788,    330,  59165,    497,    330,   7058,    929,    788,    330,
#             51,    927,  19666,   9207, 151645,    198])

# 토큰 디코딩 예시
decoded_text = processor.tokenizer.decode(batch["input_ids"][0])

print("\n디코딩된 텍스트:")
print(decoded_text)

# 디코딩된 텍스트:
# <|im_start|>system
# 당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.<|im_end|>
# <|im_start|>user
# 입력 정보:
# - name: Mr.Men Men's Charcoal White T-shirt
# - image: [image]

# 위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:
# 1) gender
# 2) masterCategory
# 3) subCategory
# 4) season
# 5) usage
# 6) baseColour
# 7) articleType

# 출력 시 **아래 JSON 예시 형태**를 반드시 지키세요:
# {
#   "gender": "예시값",
#   "masterCategory": "예시값",
#   "subCategory": "예시값",
#   "season": "예시값",
#   "usage": "예시값",
#   "baseColour": "예시값",
#   "articleType": "예시값"
# }

# # 예시
# {
#   "gender": "Men",
#   "masterCategory": "Accessories",
#   "subCategory": "Eyewear",
#   "season": "Winter",
#   "usage": "Casual",
#   "baseColour": "Blue",
#   "articleType": "Sunglasses"
# }

# # 주의
# - 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.
# <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "season": "Fall", "usage": "Casual", "baseColour": "Grey", "articleType": "Tshirts"}<|im_end|>


# 실제 loss 계산 대상 토큰만 확인
valid_label_count = (batch["labels"][0] != -100).sum().item()

print("\nloss 계산 대상 토큰 수:")
print(valid_label_count)

# loss 계산 대상 토큰 수:
# 53

label_ids = batch["labels"][0]
label_ids_for_decode = label_ids[label_ids != -100]

decoded_labels = processor.tokenizer.decode(label_ids_for_decode)

print("\nloss 계산 대상 labels 디코딩 결과:")
print(decoded_labels)

# loss 계산 대상 labels 디코딩 결과:
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "season": "Fall", "usage": "Casual", "baseColour": "Grey", "articleType": "Tshirts"}<|im_end|>

# 04_01_09_학습 설정 및 테스트 Run

TEST_RUN = False
# 먼저 2스텝(샘플 32개)만 학습해 메모리·속도 확인 → 확인 후 False

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,

#    per_device_train_batch_size=2,
    per_device_train_batch_size=1,
#    gradient_accumulation_steps=8,  # 유효 배치 16은 그대로
    gradient_accumulation_steps=16,

    gradient_checkpointing=True,
#    optim="adamw_torch_fused",
    optim="adamw_bnb_8bit",             # Adam 상태 1.06GB → 0.26GB

#    logging_steps=10,
    logging_steps=1 if TEST_RUN else 10,

    max_steps=2 if TEST_RUN else -1,   # -1이면 num_train_epochs대로 학습
    # save_strategy="steps",
    save_strategy="no" if TEST_RUN else "steps",

    save_steps=50,
    save_total_limit=3,

    bf16=True,
    fp16=False,

    learning_rate=1e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    push_to_hub=False,
    remove_unused_columns=False,

    dataset_kwargs={
        "skip_prepare_dataset": True,
    },

    report_to=None,
)

# 04_01_10_Trainer 생성

# SFTTrainer가 생성될 때 peft_config로 LoRA를 여기서 한 번 붙입니다.
# LoRA 가중치를 bf16으로 맞추고 입력 기울기도 켭니다.

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor,
)

# 학습되는 파라미터 수 확인 (r=64면 약 1.32억 개)

trainer.model.print_trainable_parameters()            # r=64면 약 1.32억 개여야 함

import time
torch.cuda.reset_peak_memory_stats(device)
start = time.perf_counter()

# trainer.train()

# 스텝마다 샘플 1개 → collate_fn → 비전 인코더가 이미지를 시각 토큰으로 변환
# → image_pad 자리에 끼움 → 텍스트 디코더(4비트 + LoRA) → 정답 위치 손실 → 역전파.
# 16번 모이면 A·B를 업데이트.

with sdpa_kernel(attn_backend):     
    trainer.train()

torch.cuda.synchronize(device)
print(f"최대 GPU 메모리: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
if TEST_RUN:
    sec = (time.perf_counter() - start) / 2
    print(f"스텝당 {sec:.1f}초 → 본 학습 552스텝 예상 약 {sec * 552 / 3600:.1f}시간")
    raise SystemExit("시험 끝: TEST_RUN = False로 바꾸고 본 학습을 실행하세요.")

trainer.save_model(f"{OUTPUT_DIR}/final")

# trainable params: 132,120,576 || all params: 4,569,936,384 || trainable%: 2.8911
# The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 151645, 'bos_token_id': None, 'pad_token_id': 151643}.
# {'loss': 1.0483, 'grad_norm': 2.765625, 'learning_rate': 0.0001, 'entropy': 0.8962185606360435, 'num_tokens': 6840.0, 'mean_token_accuracy': 0.9018514044582844, 'epoch': 0.0}
# {'loss': 0.5721, 'grad_norm': 1.96875, 'learning_rate': 0.0001, 'entropy': 0.9593606740236282, 'num_tokens': 13685.0, 'mean_token_accuracy': 0.9212490729987621, 'epoch': 0.01}
# {'train_runtime': 25.2235, 'train_samples_per_second': 1.269, 'train_steps_per_second': 0.079, 'train_loss': 0.8102031946182251, 'epoch': 0.01}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:25<00:00, 12.61s/it]
# 최대 GPU 메모리: 4.38 GB
# 스텝당 11.3초 → 본 학습 552스텝 예상 약 1.7시간
# 시험 끝: TEST_RUN = False로 바꾸고 본 학습을 실행하세요.




# 학습 뒤 병합, 업로드

# from peft import PeftModel
# from transformers import AutoProcessor, AutoModelForImageTextToText

# import torch

# adapter_path = "./output_dir_qwen3_vl_4b_instruct_lora/checkpoint-552"
# base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
# merged_path = "merged"

# # 베이스 모델 로드
# model = AutoModelForImageTextToText.from_pretrained(
#     base_model_id,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# # LoRA 어댑터 로드 및 병합
# print(f"Loading and merging PEFT from: {adapter_path}")
# peft_model = PeftModel.from_pretrained(model, adapter_path)
# merged_model = peft_model.merge_and_unload()

# merged_model.save_pretrained(
#     merged_path,
#     safe_serialization=True,
#     max_shard_size="2GB",
# )

# # 프로세서 로드
# processor = AutoProcessor.from_pretrained(base_model_id)
# processor.save_pretrained(merged_path)

# from huggingface_hub import HfApi

# api = HfApi()
# username = "iamjoon"
# MODEL_NAME = "Qwen3-VL-4B-Instruct-fashion-product-images-small-checkpoint-552"

# api.create_repo(
#     token="hf_여러분의 키 값",
#     repo_id=f"{username}/{MODEL_NAME}",
#     repo_type="model",
# )

# api.upload_folder(
#     token="hf_여러분의 키 값",
#     repo_id=f"{username}/{MODEL_NAME}",
#     folder_path="merged",
# )



