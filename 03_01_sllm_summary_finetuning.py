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
