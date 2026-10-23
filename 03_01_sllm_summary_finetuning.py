#3장 sLLM
# Chapter 3 Supervised Fine-Tuning (SFT) 및 학습 효율화

# 표 데이터 → 대화 3단 구성 → 템플릿 문자열 → 토큰화 → LoRA 학습 순서


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

torch.cuda.set_per_process_memory_fraction(0.85, device)
# 약 6.8GB 넘으면 RAM을 빌려 쓰지 않고 OOM 오류로 멈춤

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
# transformers: 4.55.2 | accelerate: 1.8.1 | trl: 0.19.0 | peft: 0.10.0 | datasets: 5.0.1
# GPU: NVIDIA GeForce RTX 5060 Laptop GPU | Compute capability: (12, 0)
# VRAM 여유/전체: 6.9 / 8.0 GB
# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'


## 03_01_01

# RunPod에서 작은 컨테이너 디스크 대신 큰 볼륨(/workspace)에 모델을 받으려는 경우
# 로컬에서는 기본 캐시(C:\Users\<사용자>\.cache\huggingface)를 그대로 쓰므로 실행하지 않는다.
# - HF_HOME: Hugging Face가 모델·데이터셋을 저장하는 최상위 폴더
# - TRANSFORMERS_CACHE: 사용 중단 예정인 변수 (HF_HOME만으로 충분)
# - DATASETS_CACHE: datasets가 읽지 않는 이름 (올바른 이름은 HF_DATASETS_CACHE)
# - 캐시 경로는 transformers·datasets를 import하기 전에 정해야 적용된다.
#   이 파일은 맨 위에서 이미 import하므로, 쓰려면 이 셀을 import보다 앞으로 옮겨야 한다.

# cache_dir = "/workspace/huggingface_cache"
# os.environ['HF_HOME'] = cache_dir
# os.environ['TRANSFORMERS_CACHE'] = cache_dir
# os.environ['DATASETS_CACHE'] = cache_dir
# os.makedirs(cache_dir, exist_ok=True)
#
# print(f" Hugging Face 캐시 디렉토리가 대용량 스토리지로 설정되었습니다:")
# print(f"{os.environ.get('HF_HOME')}")


## 03_01_02

# 1. 라이브러리 불러오기
# os·torch·pandas는 위에서 이미 import했지만 노트북 셀을 그대로 옮긴 것으로, 실행에는 영향이 없다.

# 기본 시스템 및 환경 관리:
import os
import json

# 딥러닝 프레임워크:
import torch
import accelerate

# 데이터 처리:
import pandas as pd
from datasets import Dataset

# Hugging Face 모델과 토크나이저:
from transformers import AutoTokenizer, AutoModelForCausalLM

# 파인튜닝 관련 라이브러리:
# - SFTConfig, SFTTrainer: 지도 미세조정(SFT) 설정과 학습 루프
# - LoraConfig, get_peft_model: LoRA 어댑터 설정과 모델에 붙이기
# - PeftModel: 학습한 LoRA 어댑터를 베이스 모델에 불러오기
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel


# 2. 데이터 불러오기

# 뉴스 기사와 핵심 이벤트 문장을 담은 JSON Lines 파일(한 줄에 기사 하나)을 URL에서 바로 읽는다.
# 열 구성: doc_id(문서 ID), doc_title(제목), text(본문), text_category(카테고리),
#          event_quantity(이벤트 개수), event_sentence1~10(이벤트 문장 = 학습 정답)
# 기사마다 이벤트 개수가 달라, 뒤쪽 event_sentence 열일수록 빈 값이 많다(Non-Null 수 감소).

import pandas as pd

url ='https://raw.githubusercontent.com/sogmgm/sogm_dataset/main/event_dataset_for_ft.jsonl'
df = pd.read_json(url, lines=True, encoding='utf-8')
print(len(df))

# df.info()는 표를 직접 출력하고 None을 반환하므로, 출력 마지막 줄에 None이 찍힌다.
# df['text_category'].unique()는 노트북에서는 결과가 보이지만 .py 스크립트에서는 print 없이 출력되지 않는다.
#   카테고리는 '부동산', '산업', '오피니언', '증권' 4개다.
print(df.info())
df['text_category'].unique()

# 3265
# <class 'pandas.DataFrame'>
# RangeIndex: 3265 entries, 0 to 3264
# Data columns (total 15 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   doc_id            3265 non-null   int64
#  1   doc_title         3265 non-null   str
#  2   text              3265 non-null   str
#  3   text_category     3265 non-null   str
#  4   event_quantity    3265 non-null   int64
#  5   event_sentence1   3265 non-null   str
#  6   event_sentence2   3152 non-null   str
#  7   event_sentence3   938 non-null    str
#  8   event_sentence4   310 non-null    str
#  9   event_sentence5   127 non-null    str
#  10  event_sentence6   52 non-null     str
#  11  event_sentence7   24 non-null     str
#  12  event_sentence8   9 non-null      str
#  13  event_sentence9   3 non-null      str
#  14  event_sentence10  2 non-null      str
# dtypes: int64(2), str(13)
# memory usage: 10.7 MB
# None

# 3. 데이터 샘플 확인

# 데이터 출력
# n번째 기사의 메타 정보, 본문, 이벤트 문장을 출력한다.
# 이벤트 문장은 본문에서 증감·변화가 드러나는 핵심 문장으로, 모델이 생성하도록 학습할 정답이다.
# pd.notna로 비어 있는(NaN) 이벤트 열은 건너뛴다.

n= 5 # n번째 문서

print(f"===== {n} 번째 데이터 샘플 =====")
print(f"문서 ID: {df.iloc[n]['doc_id']}")
print(f"제목: {df.iloc[n]['doc_title']}")
print(f"카테고리: {df.iloc[n]['text_category']}")
print(f"이벤트 개수: {df.iloc[n]['event_quantity']}")

print("\n===== 본문 =====")
print(df.iloc[n]['text'])

print("\n=====이벤트 문장들 =====")
for i in range(1, 11):
    event_col = f'event_sentence{i}'
    if pd.notna(df.iloc[n][event_col]):
        print(f"이벤트 {i}: {df.iloc[n][event_col]}")

# ===== 5 번째 데이터 샘플 =====
# 문서 ID: 21943
# 제목: 목동6단지 안전진단 통과…재건축 탄력
# 카테고리: 부동산
# 이벤트 개수: 3

# ===== 본문 =====
# 목동6단지가 서울 양천구 목동신시가지 2만6000여 가구 중 최초로 재건축 정밀안전진단을 조건부로 통과했다. 정밀안전진단은 재건축을 위한 첫 관문이다. 지난주 목동 재건축에서 가장 큰 걸림돌이었던 1~3단지 종상향(용적률 50% 증가) 문제도 해결돼 이 일대 재건축사업이 급물살을 탈 전망이다. 다만 이번 안전진단이 조건부 통과이고 향후 정비구역 지정, 조합 설립 및 각종 인가 등 숱한 단계가 남아 있어 본격적으로 재건축이 추진되기까지 상당한 시일이 소요될 전망이다.  31일 양천구청은 목동신시가지6단지 재건축을 위한 정밀안전진단 결과가 'D등급'이라고 목동6단지 재건축추진준비위원회에 통보했다. 재건축 안전진단의 경우 A~C등급은 유지·보수(재건축 불가), D등급은 조건부 재건축(공공기관 검증 필요), E등급은 재건축 확정 판정으로 분류된다. 목동6단지는 D등급을 받은 만큼 향후 6개월간 국토교통부 산하 한국시설안전공단의 적정성 검토를 거쳐 최종 안전진단 결과를 통보받는다.  이번에 목동6단지는 종합평가 결과 51.22점을 받아 조건부 재건축 점수 기준(30점 초과~55점 이하)을 충족했다. 지난해 10월 안전진단을 통과하지 못한 송파구 방이동 올림픽선수촌아파트(이하 올선)는 58.61점으로 재건축 불가 판정을 받았는데 이와 비교해보면 7점 넘게 낮은 점수를 받았다.    두 아파트 단지를 비교해 보면 목동6단지는 항목 4개 중 2개(주거 환경·비용 분석)에선 올선에 비해 점수가 높았지만, 나머지 2개 항목(건축 마감 및 설비노후도·구조안전성)은 점수가 낮았다. 다시 말해 목동은 주거 환경은 올선에 비해 좋지만 노후화돼 있고 구조적으로 안전하지 않다는 것이다. 신종섭 목동6단지 재건축추진준비위원장은 "목동의 경우 아파트 아래에 철근 기둥만 박고, 철근 기둥을 받치는 지지대가 없어 지진이 멀리서 나면 사람들이 흔들림을 느낄 정도"라고 설명했다. 다만 이와 관련해 '2년 더 먼저 지어진' 목동6단지가 정밀안전진단 중 가장 가중치(50%)가 높은 구조안전성 분야에서 올선보다 20점 넘게 낮은 점수(목동6단지 60.68점·올선 81.91점)를 받은 것을 두고 향후 논란의 가능성이 제기된다. 올선 재건축모임은 지난해 10월 통보된 안전진단 결과에 대해 강한 불신을 드러내고 있다.  이날 목동6단지가 정밀안전진단을 조건부로 통과하면서 목동 일대는 재건축 추진 기대감으로 들썩이는 분위기다. 특히 지난주 목동1~3단지 주민들의 숙원인 종상향(2종→3종·재건축 시 5100여 가구를 더 지을 수 있게 됨)이 서울시 심의를 통과해 사업성도 좋아졌다. 다만 공공기관의 적정성 검토가 남아 있다. 결과는 6개월 후에 나온다. 앞서 구로구 오류동 동부그린은 적정성 검토에서 1차 때 D등급을 받았는데 C등급으로 수정된 바 있다.  아울러 특별계획구역 지정도 문제다. 1~3단지는 특별계획구역에 잠정적으로 지정됐지만, 아직 4~14단지는 지정되지 못하고 있다. 특별계획구역이란 대규모 개발사업에 대해 서울시가 건축물 위치 등에 대한 세세한 지침을 주는 대신 종상향과 같은 혜택을 보는 지역이다. 서울시 관계자는 "부동산 시장 과열 여부 등을 면밀히 살펴보면서 전체 목동 단지에 대한 특별계획구역 지정을 검토할 것"이라며 시기를 조율할 것임을 시사했다.  [손동우 기자 / 나현준 기자]

# =====이벤트 문장들 =====
# 이벤트 1: 목동6단지가 서울 양천구 목동신시가지 2만6000여 가구 중 최초로 재건축 정밀안전진단을 조건부로 통과했다.
# 이벤트 2: 31일 양천구청은 목동신시가지6단지 재건축을 위한 정밀안전진단 결과가 'D등급'이라고 목동6단지 재건축추진준비위원회에 통보했다.
# 이벤트 3: 앞서 구로구 오류동 동부그린은 적정성 검토에서 1차 때 D등급을 받았는데 C등급으로 수정된 바 있다.

# 4. 학습셋·평가셋 분리

# 원래 순서의 뒤쪽에서 10%를 평가셋으로 떼어 둔다. 이 데이터는 학습에 쓰지 않는다.
# 앞쪽 순서가 그대로 유지되므로, 3.5.3의 검증 샘플(0번째) 출력은 바뀌지 않는다.
# - n_eval = int(3265 × 0.1) = 326건, 나머지 2,939건이 학습셋
# - reset_index(drop=True): 잘라낸 데이터의 인덱스를 0부터 다시 매긴다.
# - eval_split.jsonl은 03_2 평가 코드에서 다시 불러온다. force_ascii=False로 한글을 그대로 저장한다.
n_eval = int(len(df) * 0.1)
df_eval  = df.iloc[-n_eval:].reset_index(drop=True)
df_train = df.iloc[:-n_eval].reset_index(drop=True)

# 평가는 학습 뒤 별도 노트북에서 하므로 평가셋을 파일로 저장
df_eval.to_json("eval_split.jsonl", orient="records", lines=True, force_ascii=False)

print(f"train {len(df_train)} / eval {len(df_eval)}")   # 2939 / 326


## 03_01_03

import json

def create_structured_chat_data(df):
    training_data = []

    for idx, row in df.iterrows():
        # 이벤트 문장들 수집
        event_sentences = []
        for i in range(1, 11):
            event_col = f'event_sentence{i}'
            if pd.notna(row[event_col]):
                event_sentences.append(row[event_col])

        # 구조화된 시스템 프롬프트
        system_prompt = """당신은 뉴스 텍스트를 분석하여 카테고리를 분류하고 주요 핵심 이벤트들을 추출하는 전문 분석 시스템입니다.
주어진 텍스트를 분석하여 반드시 파이썬의 dictionary 형식으로 결과를 작성하십시오.
큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다.
아래 dictionary에서 각 value는 지시사항에 해당합니다. 지시사항을 그대로 적지 마시고, 해당 지시사항에 따라 적절한 value를 채워넣으십시오.

분석 결과는 다음 형식으로 작성하십시오:

답변:
{"category": "텍스트의 카테고리를 ['부동산', '산업', '오피니언', '증권'] 중 하나로 분류하여 작성하십시오",
"event_count": "텍스트 내에서 발견된 핵심 이벤트의 개수를 정수로 작성하십시오(증감, 변화, 변동이 나오는 구절을 중점적으로 보세요.)",
"events": ["텍스트에서 추출한 각 핵심 이벤트를 요약하는 문장들을 파이썬 문자열 리스트 형태로 작성하십시오. 원문의 핵심 내용을 그대로 유지하되 한 문장으로 간결하게 작성하십시오"]}"""

        # 사용자 프롬프트
        user_prompt = f"""다음 뉴스 텍스트를 분석해주세요:

{row['text']}"""

        # 어시스턴트 응답 (Dict 형식)
        events_list = [event.strip() for event in event_sentences]

        assistant_response = f"""{{"category": "{row['text_category']}","event_count": {row['event_quantity']},"events": {json.dumps(events_list, ensure_ascii=False)}}}"""

        sample = {
                      "doc_id": row['doc_id'],
                      "system_prompt": system_prompt,
                      "user_prompt": user_prompt,
                      "assistant": assistant_response
                  }

        training_data.append(sample)

    return training_data

# 데이터 함수 실행
training_data = create_structured_chat_data(df_train)

# 첫 번째 샘플 확인
print(" ===== 시스템 프롬프트  =====")
print(training_data[0]['system_prompt'])
print("\n===== 사용자 프롬프트  =====")
print(training_data[0]['user_prompt'][:200] + "...")
print("\n===== 어시스턴트 응답 =====")
print(training_data[0]['assistant'])

#  ===== 시스템 프롬프트  =====
# 당신은 뉴스 텍스트를 분석하여 카테고리를 분류하고 주요 핵심 이벤트들을 추출하는 전문 분석 시스템입니다.
# 주어진 텍스트를 분석하여 반드시 파이썬의 dictionary 형식으로 결과를 작성하십시오.
# 큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다.
# 아래 dictionary에서 각 value는 지시사항에 해당합니다. 지시사항을 그대로 적지 마시고, 해당 지시사항에 따라 적절한 value를 채워넣으십시오.

# 분석 결과는 다음 형식으로 작성하십시오:

# 답변:
# {"category": "텍스트의 카테고리를 ['부동산', '산업', '오피니언', '증권'] 중 하나로 분류하여 작성하십시오",
# "event_count": "텍스트 내에서 발견된 핵심 이벤트의 개수를 정수로 작성하십시오(증감, 변화, 변동이 나오는 구절을 중점적으로 보세요.)",
# "events": ["텍스트에서 추출한 각 핵심 이벤트를 요약하는 문장들을 파이썬 문자열 리스트 형태로 작성하십시오. 원문의 핵심 내용을 그대로 유지하되 한 문장으로 간결하게 작성하십시오"]}

# ===== 사용자 프롬프트  =====
# 다음 뉴스 텍스트를 분석해주세요:

# 대통령 선거 후 수도권을 중심으로 한 부동산시장이 조금씩 바닥을 다지며 다시 상승을 준비하는 모습이 곳곳에서 포착되고 있다. 윤석열 대통령 당선인의 재건축·재개발 공약으로 서울에서 재건축 단지가 많은 지역 아파트값은 하락폭을 줄이고 있고, 일선 부동산 공인중개소에서는 매물을 거둬들이는 이가 점차 늘고 있는 것으로 나타났다...

# ===== 어시스턴트 응답 =====
# {"category": "부동산","event_count": 2,"events": ["17일 한국부동산원에 따르면 서울의 3월 둘째 주(14일 기준) 아파트 매매가격지수는 전주 대비 0.02% 하락했다.", "경기 지역 역시 재건축 대상 아파트가 많은 고양(-0.02%→0%)이 보합으로 돌아섰고, 성남 분당은 전주와 같은 하락폭(-0.01%)을 기록했다."]}

def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": sample["system_prompt"],
            },
            {
                "role": "user",
                "content": sample["user_prompt"],
            },
            {
                "role": "assistant",
                "content": str(sample["assistant"])
            },
        ],
    }

# 전체 데이터에 format_data 적용
formatted_data = [format_data(sample) for sample in training_data]

# 확인
print("\n ===== 포맷된 데이터 샘플  =====")
# formatted_data[0]
print(formatted_data[0])

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)

print("모델과 토크나이저 로드 완료")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model parameters: {model.num_parameters():,}")

# 모델과 토크나이저 로드 완료
# Tokenizer vocab size: 151643
# Model parameters: 1,720,574,976
# 템플릿 적용 완료: 2939개 샘플
# 첫 번째 샘플 길이: 2684

# 전체 데이터에 템플릿 적용
# 학습 데이터 2,939건의 대화(system/user/assistant)를 모델이 실제로 학습할 문자열 하나로 바꾸는 작업입니다.




texts = []
for sample in formatted_data:
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt= False,
        chat_template_kwargs= {"enable_thinking": False}

    )
    texts.append(text)

print(f"템플릿 적용 완료: {len(texts)}개 샘플")
print(f"첫 번째 샘플 길이: {len(texts[0])}")
print(texts[0])

# 템플릿 적용 완료: 2939개 샘플
# 첫 번째 샘플 길이: 2684
# <|im_start|>system

# 당신은 뉴스 텍스트를 분석하여 카테고리를 분류하고 주요 핵심 이벤트들을 추출하는 전문 분석 시스템입니다.
# 주어진 텍스트를 분석하여 반드시 파이썬의 dictionary 형식으로 결과를 작성하십시오.
# 큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다.
# 아래 dictionary에서 각 value는 지시사항에 해당합니다. 지시사항을 그대로 적지 마시고, 해당 지시사항에 따라 적절한 value를 채워넣으십시오.

# 분석 결과는 다음 형식으로 작성하십시오:

# 답변:
# {"category": "텍스트의 카테고리를 ['부동산', '산업', '오피니언', '증권'] 중 하나로 분류하여 작성하십시오",
# "event_count": "텍스트 내에서 발견된 핵심 이벤트의 개수를 정수로 작성하십시오(증감, 변화, 변동이 나오는 구절을 중점적으로 보세요.)",
# "events": ["텍스트에서 추출한 각 핵심 이벤트를 요약하는 문장들을 파이썬 문자열 리스트 형태로 작성하십시오. 원문의 핵심 내용을 그대로 유지하되 한 문장으로 간결하게 작성하십시오"]}<|im_end|>
# <|im_start|>user
# 다음 뉴스 텍스트를 분석해주세요:

# 대통령 선거 후 수도권을 중심으로 한 부동산시장이 조금씩 바닥을 다지며 다시 상승을 준비하는 모습이 곳곳에서 포착되고 있다. 윤석열 대통령 당선인의 재건축·재개발 공약으로 서울에서 재건축 단지가 많은 지역 아파트값은 하락폭을 줄이고 있고, 일선 부동산 공인중개소에서는 매물을 거둬들이는 이가 점차 늘고 있는 것으로 나타났다. 다만 전문가들은 아직 부동산과 관련해 구체적인 정책들이 나오지 않았기에 매도자들은 좀 더 지켜볼 필요가 있다는 반응이다.     17일 한국부동산원에 따르면 서울의 3월 둘째 주(14일 기준) 아파트 매매가격지수는 전주 대비 0.02% 하락했다. 8주 연속 떨어졌지만 지난주(-0.02%)에 비해 하락폭이 더 커지지는 않았다. 특히 재건축 단지가 많은 노원구(-0.02%→-0.01%)와 양천구(-0.01%→0%)는 하락폭이 감소했거나 보합으로 돌아섰고, 강남구(-0.01%→0%), 송파구(-0.01%→0%), 강동구(-0.03%→-0.02%) 등 강남권도 전주에 비해 회복되는 모습이었다.   경기 지역 역시 재건축 대상 아파트가 많은 고양(-0.02%→0%)이 보합으로 돌아섰고, 성남 분당은 전주와 같은 하락폭(-0.01%)을 기록했다. 부동산원은 이천(0.27%)은 정주 여건이 양호한 창전·안흥·증포동 위주로, 파주(0.08%)는 문산읍·야동동 역세권 위주로, 일산서구(0.03%)는 대화·일산동 구축 위주로 상승했다고 설명했다. 전국(-0.02%)과 수도권(-0.03%) 아파트값 변동률은 전주와 다르지 않았고, 지방은 -0.01%에서 0%로 보합으로 전환됐다.  재건축 아파트 단지가 많은 서울 지역 공인중개사들도 대선 후 변하고 있는 분위기를 전했다. 노원구 상계동 A공인중개사 대표는 "기존에 매물을 내놨던 소유자 중 호가를 높이기 위해 일단 거둬들이는 이들이 나오고 있다"며 "대선 전에 비해 조금씩 분위기가 변하는 것 같다"고 전했다. 강남구 압구정동 B공인중개사 대표는 "재건축과 관련된 문의가 늘고 있는 것은 맞는 것 같다"면서 바쁜 모습을 보이며 전화를 끊기도 했다.   목동 신시가지 7단지 아파트 전용면적 66㎡는 최근까지 호가가 20억원 선에 눌려 있다가 대선 이후 21억원대로 상승하기도 했다.   하지만 아직 실제 매수로 이어지지는 않고 관망하는 모습이 강해 보인다.   서울부동산정보광장에 따르면 17일 현재 서울시 3월 아파트 매매 거래량은 127건으로 작년 같은 기간 거래량(3762건)에 비해 3.3%에 불과하다.   부동산원 역시 "서울은 규제 완화 기대감이 있는 재건축 단지나 한강변 인기 단지의 경우 매물이 소폭 감소하고 호가가 상승했지만 매수세로 이어지지는 않고 있다"고 밝혔다.  부동산 전문가들은 아직 구체적인 규제 완화책이 나오지 않은 만큼 매도자들은 좀 더 관망하는 한편, 매수자들은 조금 서두를 필요가 있다고 말한다.   김제경 투미부동산컨설팅 소장은 "재건축 아파트 매수와 관련한 문의가 최근 쏟아지고 있다"며 "매도자들의 경우 서두를 필요가 없지만 오는 6월 지방선거 전 또 한번 공약이 쏟아질 것을 감안하면 매수자들은 미리 움직이는 것이 좋다"고 조언했다. 고준석 제이에듀투자자문 대표 역시 "시장은 단기간에는 상승으로 돌아설 것으로 보이며 확실한 자금 계획이 있는 매수자들은 매수에 나서는 게 좋아 보인다"고 설명했다.    특히 오는 7월 말 임대차3법이 시행된 지 만 2년이 되면서 제도 시행 후 2년의 전세계약갱신청구권을 사용한 이들이 기간 만료 전 전세시장에 다시 나오면 전세와 매매시장 모두를 자극할 수 있기 때문에 매수자들은 좀 더 먼저 움직일 필요가 있다는 말도 들린다.     [박준형 기자]<|im_end|>
# <|im_start|>assistant
# <think>

# </think>

# {"category": "부동산","event_count": 2,"events": ["17일 한국부동산원에 따르면 서울의 3월 둘째 주(14일 기준) 아파트 매매가격지수는 전주 대비 0.02% 하락했다.", "경기 지역 역시 재건축 대상 아파트가 많은 고양(-0.02%→0%)이 보합으로 돌아섰고, 성남 분당은 전주와 같은 하락폭(-0.01%)을 기록했다."]}<|im_end|>


## 03_01_04
##  03에서 만든 학습용 문자열을 모델 입력(토큰 ID)으로 바꾸면서, 정답 부분만 학습하도록 표시하는 단계


def simple_tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=8192,
        padding=False,
        return_overflowing_tokens=False,
    )
    result["labels"] = []

    for i, text in enumerate(examples["text"]):
        input_ids = result["input_ids"][i] # 문자열 전체를 토큰 ID로 바꾼 것 (지시문 + 기사 + 정답)
        labels = [-100] * len(input_ids)          # ① 전체를 학습 제외로 초기화

        # 손실(오차)을 계산할 정답. ①~④ 과정으로 assistant 답변 구간만 남기고 나머지는 -100(학습 제외)으로 채웁니다.

        # 모델이 지시문과 기사를 그대로 외우는 것이 아니라 기사를 보고 답을 만드는 법만 배우게 하려는 것

        # ② 어시스턴트 시작 위치 찾기
        assistant_tokens = tokenizer.encode("<|im_start|>assistant\n",
                                            add_special_tokens=False)
        
        for j in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[j:j+len(assistant_tokens)] == assistant_tokens:
                start_idx = j + len(assistant_tokens)

                # ③ 어시스턴트 종료(<|im_end|>) 위치 찾기
                im_end_token = tokenizer.encode("<|im_end|>",
                                               add_special_tokens=False)[0]
                for k in range(start_idx, len(input_ids)):
                    if input_ids[k] == im_end_token:
                        end_idx = k + 1          # 종료 토큰 <|im_end|> 포함
                        break
                else:
                    end_idx = len(input_ids)

                # ④ 응답 구간 + 종료 토큰을 학습 대상으로 지정
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
                break

        result["labels"].append(labels)
    return result

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(simple_tokenize_function, batched=True)
print(f"토크나이징 완료: {len(tokenized_dataset)}개 샘플")

# 토크나이징 후 데이터 확인
print("===== 토크나이징 후 데이터셋 정보 =====")
print(tokenized_dataset)
print(f"컬럼: {tokenized_dataset.column_names}")
print(f"첫 번째 샘플 input_ids 길이: {len(tokenized_dataset[0]['input_ids'])}")

# 길이 분포 확인
lengths = [len(sample['input_ids']) for sample in tokenized_dataset]
print(f"토큰 길이 - 최소: {min(lengths)}, 최대: {max(lengths)}, 평균: {sum(lengths)/len(lengths):.1f}")

sample = tokenized_dataset[0]
print(f"전체 토큰 수: {len(sample['input_ids'])}")

print(f"input_ids 시작 15개: {sample['input_ids'][:15]}")
print(f"input_ids 끝 15개: {sample['input_ids'][-15:]}")

print(f"attention_mask 길이: {len(sample['attention_mask'])}")
print(f"attention_mask 시작 15개: {sample['attention_mask'][:15]}")
print(f"attention_mask 끝 15개: {sample['attention_mask'][-15:]}")
print(f"모든 값이 1인가: {all(x == 1 for x in sample['attention_mask'])}")

print(f"input_ids == labels: {sample['input_ids'] == sample['labels']}")
print(f"labels 시작 15개: {sample['labels'][:15]}")

# 학습 대상(-100이 아닌) 위치만 추리기
learn = [i for i, l in enumerate(sample['labels']) if l != -100]
print(f"학습 대상 토큰: {len(learn)}개 ({len(learn)/len(sample['labels'])*100:.1f}%)")
print(f"학습 구간: {learn[0]} ~ {learn[-1]}")
print(f"input_ids 학습 구간: {[sample['input_ids'][i] for i in learn]}")
print(f"labels 학습 구간: {[sample['labels'][i] for i in learn]}")

learn_ids = [sample['input_ids'][i] for i in learn]
print(tokenizer.decode(learn_ids, skip_special_tokens=False))

# [출력 결과]: 디코딩해 보면 <think>\n\n</think>\n\n{"category": …}<|im_end|>
# 빈 think 블록 + JSON 답변 + 종료 토큰을 학습

# 토크나이징 완료: 2939개 샘플

# ===== 토크나이징 후 데이터셋 정보 =====
# Dataset({
#     features: ['text', 'input_ids', 'attention_mask', 'labels'],
#     num_rows: 2939
# })
# 컬럼: ['text', 'input_ids', 'attention_mask', 'labels']
# 첫 번째 샘플 input_ids 길이: 1685
# 토큰 길이 - 최소: 619, 최대: 6885, 평균: 1398.7
# 전체 토큰 수: 1685
# input_ids 시작 15개: [151644, 8948, 198, 64795, 82528, 33704, 5140, 231, 112, 24897, 10764, 44104, 53189, 18411, 128618]
# input_ids 끝 15개: [135257, 4080, 15, 13, 15, 16, 11334, 17877, 54116, 49664, 128836, 1189, 13989, 151645, 198]
# attention_mask 길이: 1685
# attention_mask 시작 15개: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# attention_mask 끝 15개: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 모든 값이 1인가: True
# input_ids == labels: False
# labels 시작 15개: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
# 학습 대상 토큰: 130개 (7.7%)
# 학습 구간: 1554 ~ 1683
# input_ids 학습 구간: [151667, 271, 151668, 271, 4913, 5471, 788, 330, 63089, 57089, 85057, 2198, 3087, 3180, 788, 220, 17, 1335, 12389, 788, 4383, 16, 22, 32077, 130092, 63089, 57089, 85057, 54321, 19391, 125686, 125548, 32290, 136905, 20401, 220, 18, 128514, 5140, 239, 246, 83666, 55673, 7, 16, 19, 32077, 54116, 129044, 8, 48408, 143083, 126932, 129865, 19969, 126614, 21329, 23259, 16560, 56419, 54330, 60960, 70582, 220, 15, 13, 15, 17, 4, 53900, 132772, 128836, 10465, 330, 65306, 20487, 131937, 139512, 129242, 143483, 60960, 55902, 48408, 143083, 19969, 129875, 126429, 126345, 4080, 15, 13, 15, 17, 4, 51018, 15, 11334, 12802, 63332, 128747, 42039, 137844, 144461, 34395, 11, 128677, 131793, 128618, 64795, 33704, 56419, 54330, 80573, 129381, 53900, 132772, 135257, 4080, 15, 13, 15, 16, 11334, 17877, 54116, 49664, 128836, 1189, 13989, 151645]
# labels 학습 구간: [151667, 271, 151668, 271, 4913, 5471, 788, 330, 63089, 57089, 85057, 2198, 3087, 3180, 788, 220, 17, 1335, 12389, 788, 4383, 16, 22, 32077, 130092, 63089, 57089, 85057, 54321, 19391, 125686, 125548, 32290, 136905, 20401, 220, 18, 128514, 5140, 239, 246, 83666, 55673, 7, 16, 19, 32077, 54116, 129044, 8, 48408, 143083, 126932, 129865, 19969, 126614, 21329, 23259, 16560, 56419, 54330, 60960, 70582, 220, 15, 13, 15, 17, 4, 53900, 132772, 128836, 10465, 330, 65306, 20487, 131937, 139512, 129242, 143483, 60960, 55902, 48408, 143083, 19969, 129875, 126429, 126345, 4080, 15, 13, 15, 17, 4, 51018, 15, 11334, 12802, 63332, 128747, 42039, 137844, 144461, 34395, 11, 128677, 131793, 128618, 64795, 33704, 56419, 54330, 80573, 129381, 53900, 132772, 135257, 4080, 15, 13, 15, 16, 11334, 17877, 54116, 49664, 128836, 1189, 13989, 151645]
# <think>

# </think>

# {"category": "부동산","event_count": 2,"events": ["17일 한국부동산원에 따르면 서울의 3월 둘째 주(14일 기준) 아파트 매매가격지수는 전주 대비 0.02% 하락했다.", "경기 지역 역시 재건축 대상 아파트가 많은 고양(-0.02%→0%)이 보합으로 돌아섰고, 성남 분당은 전주와 같은 하락폭(-0.01%)을 기록했다."]}<|im_end|>


## 03_01_05
# 길이가 다른 샘플 여러 개를 하나의 배치로 묶을 때 패딩과 어텐션 마스크가 어떻게 동작하는지 보여 주는 확인

import torch

# 길이가 다른 두 샘플(0번 1,685토큰, 1번 1,546토큰)을 가져옵니다.
ex0 = tokenized_dataset[0]
ex1 = tokenized_dataset[1]
print(f"0번 데이터 길이: {len(ex0['input_ids'])}")
print(f"1번 데이터 길이: {len(ex1['input_ids'])}")

# 0번 데이터 길이: 1685
# 1번 데이터 길이: 1546

# 토크나이저의 패딩 기능으로 두 샘플을 같은 길이로 맞춰 하나의 배치로 구성
# 특별히 GPU는 같은 길이의 텐서만 한 번에 처리할 수 있어서 이 과정이 필요합니다.

batch = tokenizer.pad(
    [{"input_ids": ex0["input_ids"], "attention_mask": ex0["attention_mask"]},
     {"input_ids": ex1["input_ids"], "attention_mask": ex1["attention_mask"]}],
    padding=True,
    return_tensors="pt",
)

print("\n배치 처리 후:")
print(f"입력 ID 형태: {tuple(batch['input_ids'].shape)}")
print(f"어텐션 마스크 형태: {tuple(batch['attention_mask'].shape)}")

# 배치 처리 후:
# 입력 ID 형태: (2, 1685)
# 어텐션 마스크 형태: (2, 1685)

# 각 샘플의 마스크 합 = 실제 토큰 수, 0의 개수 = 패딩 수
print(f"0번 샘플 마스크 합(실제 토큰 수): {batch['attention_mask'][0].sum().item()}")
print(f"1번 샘플 마스크 합(실제 토큰 수): {batch['attention_mask'][1].sum().item()}")
print(f"0번 샘플 0의 개수(패딩 수): {(batch['attention_mask'][0] == 0).sum().item()}")
print(f"1번 샘플 0의 개수(패딩 수): {(batch['attention_mask'][1] == 0).sum().item()}")

# 두 샘플의 마스크가 서로 다른가?
print(f"\n두 샘플의 어텐션 마스크가 다른가? "
      f"{not torch.equal(batch['attention_mask'][0], batch['attention_mask'][1])}")

# 0번 샘플 마스크 합(실제 토큰 수): 1685
# 1번 샘플 마스크 합(실제 토큰 수): 1546

# 0번 샘플 0의 개수(패딩 수): 0
# 1번 샘플 0의 개수(패딩 수): 139

# 두 샘플의 어텐션 마스크가 다른가? True


## 03_01_06
## LoRA
## 원래 모델의 가중치 17억 개는 그대로 얼려 두고, 작은 행렬만 새로 추가해서 그 부분만 학습
## 원래 가중치 W는 고정합니다.
## 대신 작은 행렬 A(r × 입력)와 B(출력 × r)를 덧붙여 출력을 W·x + (alpha/r)·B·A·x로 만듭니다. 처음에는 B가 0으로 시작하므로, 학습 전에는 베이스 모델과 결과가 똑같습니다.


# LoRA 설정
peft_config = LoraConfig(
    r=16,                    # LoRA의 rank (낮을수록 파라미터 수 감소)
    lora_alpha=32,          # LoRA 스케일링 파라미터( 출력에 곱하는 배율(alpha/r = 32/16 = 2))
    lora_dropout=0.05,      # LoRA 레이어의 드롭아웃(과적합 방지)
    bias="none",            # bias 파라미터 학습 여부
    target_modules=["q_proj", "v_proj"],  # LoRA를 적용할 모듈
    # 어텐션의 Query와 Value 변환에만 붙입니다. K, O, MLP는 그대로 둡니다.

    task_type="CAUSAL_LM"   # 언어 모델링 태스크
)

# 모델에 LoRA 적용
model = get_peft_model(model, peft_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("LoRA 적용 완료")
print(f"학습 가능한 파라미터: {trainable_params:,}")
print(f"전체 파라미터: {total_params:,}")
print(f"학습 가능 비율: {trainable_params/total_params*100:.4f}%")

# LoRA 적용 완료
# 학습 가능한 파라미터: 3,211,264
# 전체 파라미터: 1,723,786,240
# 학습 가능 비율: 0.1863%

# 출력 숫자 검산 (Qwen3-1.7B: 28층, hidden 2048)

# q_proj(2048→2048): 16×2048 + 2048×16 = 65,536
# v_proj(2048→1024): 16×2048 + 1024×16 = 49,152
# (65,536 + 49,152) × 28층 = 3,211,264개로 출력과 정확히 일치합니다. 전체의 **0.19%**입니다.
# 전체 파라미터 1,723,786,240개는 원래 1,720,574,976개에 LoRA 3,211,264개를 더한 값입니다.

## 03_01_07

# SFT 학습 설정
# 배치 크기 1, 누적 4 (유효 배치 4는 그대로)
# 정답 위치에서만 logits 계산 (핵심 변경)

sft_cfg = SFTConfig(
    output_dir="./qwen3-lora-ft",           # 모델 저장 디렉토리
    num_train_epochs=1,                     # 학습 에포크 수

    # 배치 크기 1, 누적 4 (유효 배치 4는 그대로)

    per_device_train_batch_size=1,          # 배치 크기 < 수정 2 -> 1 >
    gradient_accumulation_steps=4,          # 그래디언트 누적 스텝 <수정 2 -> 4 >
    learning_rate=2e-4,                     # 학습률
    optim="adamw_torch",                    # 옵티마이저
    bf16=True,                              # 16비트 부동소수점 사용
    gradient_checkpointing=True,            # 메모리 절약을 위한 체크포인팅
    logging_steps=100,                      # 로그 출력 간격
    save_steps=100,                         # 모델 저장 간격
    save_total_limit=3,                     # 저장할 체크포인트 최대 개수
    max_grad_norm=0.3,                      # 그래디언트 클리핑
    warmup_ratio=0.03,                      # 웜업 비율
    lr_scheduler_type="linear",             # 학습률 스케줄러
    report_to="none",                       # 실험 추적 도구 (사용 안 함)
    remove_unused_columns=False,            # 사용하지 않는 컬럼 제거 안 함
    weight_decay=0.01,                      # 가중치 감쇠
    max_length=8192,                        # 최대 시퀀스 길이
    packing=False,                          # 시퀀스 패킹 사용 안 함
)

# SFTTrainer 초기화

import torch.nn.functional as F

class AnswerOnlyLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model()                 # PeftModel 안의 Qwen3ForCausalLM (LoRA 층 포함)

        # ① 28층만 통과시켜 마지막 은닉 상태를 얻음 (lm_head는 아직 적용 안 함)
        hidden = base.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=False).last_hidden_state      # (1, 길이, 2048)

        # i번째 위치가 i+1번째 토큰을 맞히므로 정답을 한 칸 당김
        shift_labels = inputs["labels"][:, 1:]
        mask = shift_labels != -100                                 # 정답 위치만 True

        # ② 정답 위치의 은닉 상태만 골라 lm_head 적용 → (정답 토큰 수, 151936)
        logits = base.lm_head(hidden[:, :-1][mask]).float()

        # ③ 누적 4스텝의 정답 토큰 수로 나눠 평균 (책 방식과 같은 기준)
        loss = F.cross_entropy(logits, shift_labels[mask], reduction="sum")
        loss = loss / (num_items_in_batch if num_items_in_batch is not None else mask.sum())

        # (선택) 정답 토큰 정확도 로그: 책 방식의 mean_token_accuracy와 같은 지표
        self._metrics["train"]["mean_token_accuracy"].append(
            (logits.argmax(-1) == shift_labels[mask]).float().mean().item())
        return (loss, None) if return_outputs else loss


# 본 학습 전에 가장 긴 샘플로 메모리·속도 시험 (확인 후 False로 바꿔 본 학습)
# 가장 긴 샘플 8개로 2스텝(배치 1 × 누적 4 × 2)만 학습해 최대 메모리와 속도를 잽니다.
# 시험이 끝나면 프로그램을 종료하므로 본 학습에는 영향이 없습니다.
# TEST_RUN = True로 한 번 실행해 결과를 확인한 뒤, False로 바꿔 본 학습을 실행하면 됩니다. 

print("어텐션 방식:", model.config._attn_implementation)       # sdpa가 정상
print("가장 긴 8개:", sorted(lengths, reverse=True)[:8])
for t in (3072, 4096):
    print(f"{t}토큰 초과: {sum(l > t for l in lengths)}개 / {len(lengths)}개")

# 어텐션 방식: sdpa
# 가장 긴 8개: [6885, 4271, 4261, 4097, 4041, 3998, 3961, 3835]
# 3072토큰 초과: 25개 / 2939개
# 4096토큰 초과: 4개 / 2939개

TEST_RUN = False

if TEST_RUN:
    import time
    # 03_01_04의 lengths(샘플별 토큰 수)로 가장 긴 8개 선택
    longest = sorted(range(len(tokenized_dataset)), key=lambda i: lengths[i], reverse=True)[:8]
    test_cfg = SFTConfig(
        output_dir="./qwen3-lora-test",
        max_steps=2,                        # 2스텝 × 누적 4 = 샘플 8개
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",                 # 체크포인트 저장 안 함
        report_to="none",
        remove_unused_columns=False,
        max_length=8192,
        packing=False,
    )
    test_trainer = AnswerOnlyLossTrainer(model=model, args=test_cfg,
                                         train_dataset=tokenized_dataset.select(longest),
                                         processing_class=tokenizer)
    model.enable_input_require_grads()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    # test_trainer.train()
    with sdpa_kernel(attn_backend):
        test_trainer.train()
    torch.cuda.synchronize(device)
    sec_per_step = (time.perf_counter() - start) / 2
    print(f"가장 긴 샘플 기준 최대 GPU 메모리: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    print(f"스텝당 {sec_per_step:.1f}초 → 전체 예상 약 {sec_per_step * 0.2 * 735 / 60:.0f}분")
    raise SystemExit("시험 끝: TEST_RUN = False로 바꾸고 본 학습을 실행하세요.")

# 가장 긴 샘플 기준 최대 GPU 메모리: 4.84 GB
# 스텝당 8.9초 → 전체 예상 약 22분
# 시험 끝: TEST_RUN = False로 바꾸고 본 학습을 실행하세요.

# trainer = SFTTrainer(
trainer = AnswerOnlyLossTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    processing_class=tokenizer,
)




if not os.path.exists("./qwen3-lora-ft-final"):    # 학습 결과가 없을 때만 학습

    # 모델의 입력 그래디언트 활성화 (LoRA 학습에 필요)
    model.enable_input_require_grads()

    # 데이터셋 셔플 및 학습 시작
    trainer.train_dataset = trainer.train_dataset.shuffle(seed=42)
    # trainer.train()
    with sdpa_kernel(attn_backend):
        trainer.train()

    # 학습 완료 후 모델 저장
    trainer.save_model("./qwen3-lora-ft-final")
    print("훈련 완료 모델 저장됨.")

# {'loss': 0.1687, 'grad_norm': 0.416015625, 'learning_rate': 0.00017865168539325842, 'mean_token_accuracy': 0.9696124362945556, 'epoch': 0.14}
# {'loss': 0.0671, 'grad_norm': 0.2353515625, 'learning_rate': 0.000150561797752809, 'mean_token_accuracy': 0.9820817258954048, 'epoch': 0.27}
# {'loss': 0.0658, 'grad_norm': 0.22265625, 'learning_rate': 0.00012247191011235955, 'mean_token_accuracy': 0.9812659087777138, 'epoch': 0.41}
# {'loss': 0.0612, 'grad_norm': 0.234375, 'learning_rate': 9.438202247191012e-05, 'mean_token_accuracy': 0.9826123650372028, 'epoch': 0.54}
# {'loss': 0.059, 'grad_norm': 0.197265625, 'learning_rate': 6.629213483146067e-05, 'mean_token_accuracy': 0.9827923990786076, 'epoch': 0.68}
# {'loss': 0.0585, 'grad_norm': 0.359375, 'learning_rate': 3.8202247191011236e-05, 'mean_token_accuracy': 0.982940639257431, 'epoch': 0.82}
# {'loss': 0.0596, 'grad_norm': 0.1689453125, 'learning_rate': 1.0112359550561798e-05, 'mean_token_accuracy': 0.9823568019270897, 'epoch': 0.95}
# {'train_runtime': 1771.6498, 'train_samples_per_second': 1.659, 'train_steps_per_second': 0.415, 'train_loss': 0.07604663891046226, 'mean_token_accuracy': 0.9839962862378402, 'epoch': 1.0}
# 훈련 완료 모델 저장됨.



## 03_01_08  학습 결과를 확인하기 위해 모델을 다시 불러오는 준비 작업

import gc

# 학습에 쓴 트레이너·모델을 지워 GPU 메모리를 비운다 (학습 직후 같은 실행에서 이어질 때 필요)
for name in ("trainer", "model"):
    globals().pop(name, None)
gc.collect()
torch.cuda.empty_cache()


# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# 베이스 모델 로드
# base = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen3-1.7B",
#     torch_dtype="auto",
#     device_map="auto"
# ).eval()

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype="auto",
    device_map=device
).eval()

# LoRA 파인튜닝된 모델 로드: 같은 베이스에 어댑터만 붙인다 (두 번째 복사본을 만들지 않음)
# lora = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen3-1.7B",
#     torch_dtype="auto",
#     device_map="auto"
# )
# lora = PeftModel.from_pretrained(
#     lora,
#     "./qwen3-lora-ft-final",
#     device_map="auto",
#     torch_dtype="auto"
# ).eval()
lora = PeftModel.from_pretrained(base, "./qwen3-lora-ft-final").eval()


## 03_01_09

# 학습 전후 비교 테스트
# 학습에 쓰지 않은 새 뉴스 기사 3개(부동산, 산업, 증권)를 베이스 모델과 LoRA 모델에 똑같이 넣고, 출력 JSON을 나란히 비교


# 시스템 프롬프트 
system_prompt = """당신은 뉴스 텍스트를 분석하여 카테고리를 분류하고 주요 핵심 이벤트들을 추출하는 전문 분석 시스템입니다.
주어진 텍스트를 분석하여 반드시 파이썬의 dictionary 형식으로 결과를 작성하십시오.
큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다.
아래 dictionary에서 각 value는 지시사항에 해당합니다. 지시사항을 그대로 적지 마시고, 해당 지시사항에 따라 적절한 value를 채워넣으십시오.

분석 결과는 다음 형식으로 작성하십시오:

답변:
{"category": "텍스트의 카테고리를 ['부동산', '산업', '오피니언', '증권'] 중 하나로 분류하여 작성하십시오",
"event_count": "텍스트 내에서 발견된 핵심 이벤트의 개수를 정수로 작성하십시오(증감, 변화, 변동이 나오는 구절을 중점적으로 보세요.)",
"events": ["텍스트에서 추출한 각 핵심 이벤트를 요약하는 문장들을 파이썬 문자열 리스트 형태로 작성하십시오. 원문의 핵심 내용을 그대로 유지하되 한 문장으로 간결하게 작성하십시오"]}"""

# 테스트 데이터 3개 
test_texts = [
    # 1. 부동산 카테고리
    """ 최대 수혜지역이라고요? 분위기 좀 달라지긴 했지만, 그게 꼭 6·27 대출 규제 때문만은 아니죠."

지난 19일 서울 도봉구 창동역 인근 한 공인중개사는 최근 시장 분위기를 묻는 질문에 이같이 말했다. 도봉구는 창동 차량기지 복합개발, GTX-C 등 대형 교통 호재에 힘입어 올해 들어 아파트값이 상승세로 전환됐고, 일부 단지에서는 실거래가 회복 조짐도 나타나고 있다. 반면 6·27대책 발표 직후 상승세 강했던 금관구(금천·관악·구로)는 매수·매도인 모두 지켜보기에 들어간 모습이다.

■도봉구 상승세…실거래가 회복 조짐

6·27 대출 규제 시행 3주차에 접어들면서 서울 외곽 지역을 중심으로 시장의 온도차가 뚜렷해지고 있다. 한국부동산원 주간 아파트가격 동향에 따르면 7월 2주(15일 기준) 서울 아파트값은 0.19% 올라 전주(0.29%)보다 오름폭이 둔화됐다. 다만 도봉구(0.05%→0.06%)와 중구(0.16%→0.18%)는 상승폭이 소폭 확대됐다. 도봉구의 한 중개업소 관계자는 "급매가 거의 소진된 상태에서 전고점에 근접한 거래도 하나둘 나타나고 있다"며 "전용 84㎡, 165㎡ 등 다양한 평형에 실수요자와 일부 투자자의 문의가 이어지고 있다"고 전했다. 이어 "6·27 대책 이후 도봉구처럼 상대적으로 대출 여건이 유지되는 중저가 지역에 관심이 모이고 있는 건 사실이지만, 지금 시장을 움직이는 건 결국 개발 기대감에 기반한 유동성 있는 수요자들"이라고 설명했다.

서울부동산정보광장에 따르면 도봉구 아파트 거래량은 5월 155건, 6월 289건으로 늘었지만, 7월 들어서는 20일 기준 51건에 머물고 있다. 규제 여파로 거래량은 다소 숨고르기에 들어간 양상이지만 가격 상승세는 계속될 것으로 전망됐다.

도봉구의 또 다른 중개업소 대표는 "이미 작년에 바닥을 찍었다는 인식이 확산되면서 실수요자 문의가 늘고 있고, 호가도 빠르게 회복되고 있다"며 "GTX-C, 창동 차량기지 개발 등 향후 호재가 여전한 만큼, 지금은 회복의 시작일 뿐이라는 얘기도 나온다"고 했다.

■금관구 숨고르기…거래량 감소 뚜렷

반면 6·27 대책 직후 풍선효과 기대감이 부각됐던 '금관구'는 오름세가 한풀 꺾인 분위기다. 같은 기간 금천구는 0.09%→0.07%, 관악구는 0.19%→0.15%, 구로구는 0.18%→0.12%로 상승폭이 일제히 축소됐다.

금천구 시흥동의 한 중개업소 관계자는 "호가는 잠깐 반짝했지만 지금은 문의가 다시 줄어든 상황"이라며 "갭투자자든 실수요자든 모두 관망 중"이라고 말했다. 관악구의 한 공인중개사도 "전세보증금이 낮은 구축 소형 위주로 간헐적인 갭투자 수요는 있었지만, 지금은 전세금도 안 오르고 매수도 끊긴 상태"라고 분위기를 전했다.

서울부동산정보광장에 따르면 금천구 아파트 거래량은 5월 74건에서 6월 135건으로 증가한 뒤, 7월(20일 기준)에는 25건에 그쳤다. 관악구는 같은 기간 222건→294건→54건, 구로구는 678건→462건→84건으로 줄며 거래 감소세가 뚜렷하다.

개봉동의 중개업소 관계자는 "급매라고는 해도 1000만~2000만원 낮춘 수준인데, 그런 물건도 상반기에 이미 빠졌다"며 "매수자는 관망세, 매도자는 버티기에 들어간 상황"이라고 전했다. 결론적으로 매물 자체가 줄어들면서 거래량도 감소한 위축된 시장 흐름을 보이고 있어, 풍선효과가 나타난다고 보기는 어렵다는 평가다.

부동산업계 한 전문가는 "서울 외곽 지역은 규제 변화에 민감하게 반응하는 편이지만, 신축 비중이나 입지 여건, 실수요자의 선호도에 따라 시장 흐름은 엇갈릴 수밖에 없다"며 "현재는 거래 절벽에 가까운 상태로 '풍선효과' 역시 일부 지역에 국한된 단기 흐름에 그칠 가능성이 높다"고 진단했다.",
""",    
    # 2. 산업 카테고리
    """
    한일시멘트가 자회사 한일현대시멘트 흡수 합병을 통해 경영효율화에 나서면서 시멘트업계에 구조·사업개편 가능성이 제기된다. 국내 시멘트 산업이 '저성장 고비용'의 구조적 위험에 빠진데 따른 영향이라는 분석이다. 다만 한일시멘트를 제외한 주요 기업들은 우선 신중모드로 접근할 전망이다.

20일 관련 업계에 따르면 한일시멘트는 최근 이사회를 열고 77.78% 지분을 보유한 한일현대시멘트를 오는 11월 1일부로 흡수 합병하기로 의결했다. 양사는 2017년부터 단계적으로 통합 작업을 진행해왔다. 이번 합병으로 상장사 이중 구조를 해소하고 중복투자·외부비용 절감을 통한 경영 효율화에 나선다는 방침이다. 한일시멘트 관계자는 "경기가 부침이 있는 만큼 설비 등 인프라 활용해 탄력적 대응이 가능할 것"이라고 말했다.

이미 시멘트 업계의 경영상황은 악화일로다. 국토교통부에 따르면 지난 5월 건설착공면적은 전월 대비 26.9% 감소한 620만㎡ 수준에 그쳤다. 착공면적은 통상 건설경기의 선행지표로 여겨진다. 시멘트 출하량도 급감했다. 올해 1·4분기 시멘트 내수는 전년 동기 대비 21.8% 줄어든 812만t으로, 최근 5년 내 최저치를 기록했다. 2023년 1·4분기 1201만t 대비 2년 새 32.4%가 줄어든 셈이다.

그럼에도 업계 전반의 대응은 신중한 모습이다. 한일을 제외한 주요 업체들은 합병 혹은 조직개편 계획 여부 등은 없다는 입장이다. 쌍용C&E 관계자는 "경기가 좋지 않지만 수출 덕분에 그나마 나은 상황으로 당분간 특별한 계획 없다"고 말했다. 성신양회, 한라시멘트 관계자 역시 "거래처 및 비용 절감에 집중하고 있을 뿐, 지배구조 개편 등은 고려하고 있지 않다"고 했다.

업계는 현재 위기를 수요 부진과 고정비 부담의 이중고로 보고 있다. 시멘트 산업은 설비 비중이 높아 수요가 줄어도 일정 수준의 가동과 인력 유지를 지속해야 하기 때문이다. 게다가 최근 몇 년간 산업용 전기요금 인상, 탄소배출권 비용 증가, 폐기물 처리단가 상승 등 외부 비용도 커졌다.

이 가운데 업계는 환경·사회·지배구조(ESG) 전환에도 동시에 대응 중이다. 시멘트 제조 시 온실가스 배출량이 많다는 점에서, 탄소중립 압력이 거세져 온 탓이다. 이에 오는 2030년까지 2018년 대비 12%, 2050년까지 53%의 온실가스 감축을 목표로 설정하고 있다. 이를 위해 석회석 대신 비탄산염 원료를 활용한 저탄소 클링커 기술, 폐합성수지·바이오매스 등 순환자원 연료 전환 기술 개발 등에 집중하고 있다.

한일시멘트의 이번 결정은 불황기의 생존 해법으로 지배구조 단순화 및 비용 최적화의 가능성을 제기했다는 분석이 나온다.

업계 관계자는 "내수 의존도가 큰 기업일수록 버틸 수 있는 시간이 길지 않다"며 "누가 먼저 구조를 바꾸고, 얼마나 오래버티느냐가 관건"이라고 말했다.
    """,
    
    # 3. 증권 카테고리
    """ (서울=연합뉴스) 임화섭 기자 = 인공지능(AI) 분야 경쟁으로 '매그니피센트 세븐'(미국의 7대 테크 대기업·The Magnificent Seven) 사이에서도 주가 차별화가 심화되고 있다고 미국 일간 월스트리트저널(WSJ)이 20일(현지시간) 분석했다.

테크업계를 주름잡는 대기업들이며 각종 주가지수에서도 큰 비중을 차지하는 아마존닷컴(이하 아마존), 알파벳, 애플, 메타플랫폼스(이하 메타), 마이크로소프트(이하 MS), 엔비디아, 테슬라 등 7개 기업의 주가 흐름이 AI 분야 성과에 따라 크게 갈리는 경향이 최근 뚜렷해졌다는 것이다.

대표적인 주가지수인 S&P 500에서 이들 7개 업체가 차지하는 비중은 35%에 이른다.

최근 모건스탠리 보고서에 따르면 이 업체들의 2분기 순이익은 전년 동기 대비 14% 증가할 것으로 전망되며, 이는 나머지 493개 S&P 500 기업의 순이익이 전년 동기 대비 3% 감소할 것으로 예상되는 전반적 증시 분위기와 비교하면 훨씬 좋다.

이들 중 알파벳을 제외한 6개 업체들은 최근 주가가 향후 1년간 예상 순이익의 25배를 넘었으며, 이는 최근 S&P 500 기업의 평균 주가수익비율(PER) 22.35보다 높다.

올해 들어 엔비디아, 메타, MS의 주가는 약 20% 이상 올랐으나, 이와 대조적으로 애플은 16% 하락했고 알파벳도 2% 내렸고 테슬라는 18%가 빠졌다.

AI 스타트업 '앤스로픽'에 투자한 아마존은 관세 등의 영향으로 전자상거래 부문 사업에 타격이 상당히 큰 가운데서도 3% 올랐다.

알파벳과 테슬라는 오는 23일에, 메타, MS, 애플은 그 다음 주에 분기 실적을 발표할 예정이다.

해리스 파이낸셜 그룹의 매니징 파트너인 제이미 콕스는 WSJ에 "(테크 대기업들 사이에서 실적에 따라 주가 흐름이 차별화되는 것은) 불가피한 일이었다. 다른 일들을 하고 있기 때문에 영원히 서로 똑같은 (주가) 흐름이 계속되는 것은 불가능하다"며 "이제 승자와 패자의 계층분화가 왔다"고 평가했다.

2023년에 이 7개 업체를 묶어서 '매그니피센트 세븐'이라는 표현을 쓰기 시작한 뱅크오브아메리카 글로벌 리서치의 투자전략담당 책임자 마이클 하트넛은 이들 업체를 묶어서 본 중요한 이유 중 하나는 이들이 AI 분야의 선봉에 있기 때문이라고 말한 바 있다.

그러나 투자회사 스피어의 창립자 겸 투자책임자인 이바나 델레브스카는 "지금은 펀더멘털에서 (7개 업체들 사이에) 상당히 큰 차이가 보이고 있다"고 말했다.
    """
]
def generate_output(model, text, system_prompt):
    # messages 형식으로 프롬프트 구성 (system + user)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 뉴스 텍스트를 분석해주세요:\n\n{text}"}
    ]
    
    # apply_chat_template 사용 (thinking 끄기: enable_thinking=False)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # thinking 모드 OFF (여기서 변경!)
    )
    
    # 입력 토큰화
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    
    # generate (deterministic으로 설정)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            # max_length=8192,  # 최대 길이
            max_new_tokens=1024,   # 새로 생성할 최대 토큰 수
            do_sample=False   # 샘플링 off
        )
    
    # 출력 디코딩 (thinking이 없을 테니 전체 출력)
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]  # 생성된 부분만
    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # 만약 thinking 태그가 남아 있으면 파싱 (안전장치, False로 하면 불필요)
    try:
        # </think> 토큰 ID (Qwen 공식: 151668, 모델에 따라 확인 필요)
        think_end_token = 151668  # Qwen의 </think> 토큰 ID
        index = len(output_ids.tolist()) - output_ids.tolist()[::-1].index(think_end_token)
    except ValueError:
        index = 0  # thinking 없으면 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    
    # thinking이 있으면 출력, 없으면 content만
    if thinking_content:
        print("주의: thinking content가 남아 있음:", thinking_content)
    
    return content  # 최종 content 반환 (dictionary 예상)


# 5. 3개 데이터로 베이스 모델 generate 실행



for i, text in enumerate(test_texts, 1):
    # print(f"\n===== 테스트 데이터 {i} (카테고리 힌트: 부동산/산업/증권) =====\n{text}\n")
    
    # generate 호출 (모델 로드 후 바로 여기서 generate)

    # base_output = generate_output(base, text, system_prompt)
    with lora.disable_adapter():                       # LoRA를 끈 상태 = 베이스 모델
        base_output = generate_output(lora, text, system_prompt)
    print("----- 베이스 모델 출력 -----\n", base_output)

    
    # generate 호출 (모델 로드 후 바로 여기서 generate)
    # base_output = generate_output(lora, text, system_prompt)

    lora_output = generate_output(lora, text, system_prompt)

    print("----- 로라 모델 출력 -----\n", lora_output)
    
    print(f"\n===== {i}번째 비교 테스트 끝 =====\n")

# ----- 베이스 모델 출력 -----
#  {"category": "산업", "event_count": 4, "events": ["6·27 대출 규제 시행 후 도봉구 아파트 가격 상승세 유지", "도봉구 거래량은 7월 20일 기준 51건에 머물고 있다", "금관구 아파트 거래량은 7월 20일 기준 25건에 머물고 있다", "부동산업계 전문가가 서울 외곽 지역 시장 흐름에 대한 진단을 발표"]}
# The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# ----- 로라 모델 출력 -----
#  {"category": "부동산","event_count": 2,"events": ["서울부동산정보광장에 따르면 도봉구 아파트 거래량은 5월 155건, 6월 289건으로 늘었지만, 7월 들어서는 20일 기준 51건에 머물고 있다.", "서울부동산정보광장에 따르면 금천구 아파트 거래량은 5월 74건에서 6월 135건으로 증가한 뒤, 7월(20일 기준)에는 25건에 그쳤다."]}

# ===== 1번째 비교 테스트 끝 =====

# ----- 베이스 모델 출력 -----
#  답변:
# {"category": "산업", "event_count": 4, "events": ["한일시멘트가 자회사 한일현대시멘트를 흡수 합병 결정", "시멘트 산업의 경영상황 악화", "업계 전반의 대응은 신중한 모습", "시멘트 제조 시 온실가스 배출량 증가로 탄소중립 목표 설정"]}
# The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

# ----- 로라 모델 출력 -----
#  {"category": "산업","event_count": 2,"events": ["20일 관련 업계에 따르면 한일시멘트는 최근 이사회를 열고 77.78% 지분을 보유한 한일현대시멘트를 오는 11월 1일부로 흡수 합병하기로 의결했다.", "이 가운데 업계는 환경·사회·지배구조(ESG) 전환에도 동시에 대응 중이다."]}

# ===== 2번째 비교 테스트 끝 =====

# ----- 베이스 모델 출력 -----
#  답변:
# {"category": "산업", "event_count": 10, "events": ["인공지능(AI) 분야 경쟁이 '매그니피센트 세븐' 기업 주가에 심화되며, 주가 차별화가 뚜렷해졌습니다.", "Amazon, Alphabet, Apple, Meta, Microsoft, NVIDIA, Tesla의 주가가 AI 성과에 따라 크게 갈리고 있습니다.", "S&P 500에서 7개 기업이 차지하는 비중은 35%에 달하며, 2분기 순이익이 전년 동기 대비 14% 증가했습니다.", "Alphabet과 Tesla는 주가가 20% 이상 상승했으나, Apple은 16% 하락했고, Alphabet은 2% 내렸습니다.", "Amazon은 관세 영향으로 전자상거래 부문에 타격을 받았지만, 3% 올랐습니다.", "Alphabet과 Tesla는 23일 분기 실적 발표를, Meta, Microsoft, Apple는 다음 주에 발표할 예정입니다.", "이바나 델레브스카는 7개 기업이 AI 분야 선봉에 있는 이유를 강조하며, 펀더멘털 차이가 뚜렷하다고 평가했습니다.", "Harry Ferguson의 제이미 콕스는 테크 대기업 간 실적 차이로 주가 흐름이 차별화된다고 평가했습니다.", "NVIDIA는 주가가 20% 이상 상승했으나, Apple은 16% 하락했고, Tesla는 18% 하락했습니다."]}

# {"category": "산업", "event_count": 10, "events": ["인공지능(AI) 분야 경쟁이 '매그니피센트 세븐' 기업 주가에 심화되며, 주가 차별화가 뚜렷해졌습니다.", "Amazon, Alphabet, Apple, Meta, Microsoft, NVIDIA, Tesla의 주가가 AI 성과에 따라 크게 갈리고 있습니다.", "S&P 500에서 7개 기업이 차지하는 비중은 35%에 달하며, 2분기 순이익이 전년 동기 대비 14% 증가했습니다.", "Alphabet과 Tesla는 주가가 20% 이상 상승했으나, Apple은 16% 하락했고, Alphabet은 2% 내렸습니다.", "Amazon은 관세 영향으로 전자상거래 부문에 타격을 받았지만, 3% 올랐습니다.", "Alphabet과 Tesla는 23일 분기 실적 발표를, Meta, Microsoft, Apple는 다음 주에 발표할 예정입니다.", "이바나 델레브스카는 7개 기업이 AI 분야 선봉에 있는 이유를 강조하며, 펀더멘털 차이가 뚜렷하다고 평가했습니다.", "Harry Ferguson의 제이미 콕스는 테크 대기업 간 실적 차이로 주가 흐름이 차별화된다고 평가했습니다.", "NVIDIA는 주가가 20% 이상 상승했으나, Apple은 16% 하락했고, Tesla는 18% 하락했습니다."]}

# {"category": "산업", "event_count": 10, "events": ["인공지능(AI) 분야 경쟁이 '매그니피센트 세븐' 기업 주가에 심화되며, 주가 차별화가 뚜렷해졌습니다.", "Amazon, Alphabet, Apple, Meta, Microsoft, NVIDIA, Tesla의 주가가 AI 성과에 따라 크게 갈리고 있습니다.", "S&P 500에서 7개 기업이 차지하는 비중은 35%에 달하며, 2분기 순이익이 전년 동기 대비 14% 증가했습니다.", "Alphabet과 Tesla는 주가가 20% 이상 상승했으나, Apple은 16% 하락했고, Alphabet은 2% 내렸습니다.", "Amazon은 관세 영향으로 전자상거래 부문에 타격을 받았지만, 3% 올랐습니다.", "Alphabet과 Tesla는 23일 분기 실적 발표를, Meta, Microsoft, Apple는 다음 주에 발표할 예정입니다.", "이바나 델레브스
# The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# ----- 로라 모델 출력 -----
#  {"category": "증권","event_count": 2,"events": ["올해 들어 엔비디아, 메타, MS의 주가는 약 20% 이상 올랐으나, 이와 대조적으로 애플은 16% 하락했고 알파벳도 2% 내렸고 테슬라는 18%가 빠졌다.", "AI 스타트업 '앤스로픽'에 투자한 아마존은 관세 등의 영향으로 전자상거래 부문 사업에 타격이 상당히 큰 가운데서도 3% 올랐다."]}

# ===== 3번째 비교 테스트 끝 =====

# 베이스 모델은 기사 내용과 관계없이 거의 다 "산업"으로 답합니다. LoRA 모델은 세 개 모두 맞혔습니다.
# 베이스 모델: 프롬프트에 있던 답변:을 따라 쓰고, 3번 기사에서는 JSON을 반복 생성했습니다. 09에서 max_new_tokens=1024로 바꾼 이유가 이런 반복 때문입니다.
# LoRA 모델: 학습 데이터와 똑같은 형식("category": "부동산","event_count": 2,..., 쉼표 뒤 공백 없음)으로 JSON만 출력합니다. 03_2 평가에서 JSON을 읽어 들이기(파싱) 쉬워집니다.

# 이벤트 문장
# 베이스 모델: "전문가가 진단을 발표"처럼 요약하거나 고쳐 쓰고, 수치 변화가 아닌 내용도 넣습니다.
# LoRA 모델: 원문 문장을 그대로 뽑고, 수치의 증감이 있는 문장(거래량 155건→289건, 주가 20% 상승·16% 하락)을 고릅니다. 데이터셋의 정답 기준(증감·변화·변동 문장)을 그대로 배운 것입니다.

# 주의할 점
## 이벤트 개수가 늘 2개입니다. 학습 데이터에서 이벤트가 2개인 기사가 대부분이라(3개 이상은 약 29%), 모델이 2개로 치우쳐 있을 수 있습니다. 이벤트가 더 많은 기사에서는 덜 뽑을 가능성이 있습니다.
## 샘플이 3개뿐입니다. 정확한 판단은 03_2 평가에서 합니다. 학습에 쓰지 않은 326건으로 파싱 성공률, 카테고리 정확도, 개수 정확도, ROUGE를 수치로 비교합니다. 참고로 책의 결과는 카테고리 정확도 0.08 → 0.89, 개수 정확도 0.06 → 0.64였습니다.



## 03_01_10

# 토큰은 코드에 쓰지 않고, 터미널에서 `uv run hf auth login`으로 저장한 토큰을 쓴다 (README 참고)

# from huggingface_hub import login
from huggingface_hub import whoami

# HF_TOKEN = "your_huggingface_token_here"  # 본인의 토큰으로 대체

# repo_id = "sogm1/qwen3-1.7b-news-event-merged"  # 원하는 레포지토리명

hf_user = whoami()["name"]                                   # 로그인이 안 돼 있으면 여기서 오류
# repo_id = f"{hf_user}/qwen3-1.7b-news-event-merged"         # 내 계정 아래에 만든다
# merged_dir = "./qwen3-merged"  # 병합 모델을 저장할 로컬 폴더
repo_id = f"{hf_user}/qwen3-1.7b-news-event-lora"           # 어댑터만 올리는 저장소
adapter_dir = "./qwen3-lora-ft-final"                       # 03_01_07에서 저장한 LoRA 어댑터 폴더

# Hugging Face에 로그인
# print("허깅페이스에 로그인 중...")
# login(token=HF_TOKEN)
# print("로그인 성공!")
print("로그인 계정:", hf_user, "| 업로드할 저장소:", repo_id)

# 로그인 계정: HaloJEon | 업로드할 저장소: HaloJEon/qwen3-1.7b-news-event-merged

# 레포지토리 생성
from huggingface_hub import create_repo

print(f"레포지토리 {repo_id} 생성 중...")
try:
    create_repo(
        repo_id=repo_id,
        repo_type="model",  # 모델 레포지토리로 지정
        # private=False,      # 공개 레포지토리 (True로 설정하면 비공개)
        private=True,         # 비공개 (뉴스 기사로 학습한 연습용 어댑터)
        exist_ok=True       # 이미 존재하면 무시
    )
    print(f"레포지토리 생성 완료: https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"레포지토리 생성 중 오류: {e}")

# 병합 모델(약 3.4GB) 대신 어댑터만 올리므로 병합·로컬 저장은 하지 않는다.

# merged_model = lora.merge_and_unload()
# print("모델 병합 완료!")
#
# os.makedirs(merged_dir, exist_ok=True)
#
#
# # 모델과 토크나이저를 로컬에 저장
# merged_model.save_pretrained(merged_dir)
# tokenizer.save_pretrained(merged_dir)
# print("로컬 저장 완료!")

# Hugging Face Hub에 업로드
from huggingface_hub import upload_folder
from huggingface_hub import constants as hf_constants

# HF_HOME이 Windows 드라이브(/mnt/c)에 있으면 Xet 전송이 파일 오류로 실패하므로 일반 HTTP로 올린다
hf_constants.HF_HUB_DISABLE_XET = True

try:
    upload_folder(
        # folder_path=merged_dir,
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
        # commit_message="Upload merged Qwen3-1.7B model fine-tuned for Korean news event extraction"
        commit_message="Upload Qwen3-1.7B LoRA adapter fine-tuned for Korean news event extraction"
    )

    print("업로드 완료!")
    print(f"모델 URL: https://huggingface.co/{repo_id}")
    # print(f"로컬 경로: {merged_dir}")
    print(f"로컬 경로: {adapter_dir}")
    
except Exception as e:
    print(f"업로드 중 오류 발생: {e}")

# 업로드 완료!
# 모델 URL: https://huggingface.co/HaloJEon/qwen3-1.7b-news-event-lora
# 로컬 경로: ./qwen3-lora-ft-final

