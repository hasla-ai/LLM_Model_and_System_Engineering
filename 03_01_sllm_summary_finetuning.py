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
