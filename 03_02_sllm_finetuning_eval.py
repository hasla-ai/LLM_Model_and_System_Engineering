# 03_02_01

import json, ast
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from mecab import MeCab
from rouge import Rouge

# INFO 10-23 09:36:01 [__init__.py:216] Automatically detected platform cuda.
# vLLM message


# 학습 때 저장한 held-out 평가셋(학습에 쓰이지 않은 데이터)
df_eval = pd.read_json("eval_split.jsonl", lines=True, encoding="utf-8")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

system_prompt = """당신은 뉴스 텍스트를 분석하여 카테고리를 분류하고 주요 핵심 이벤트들을 추출하는 전문 분석 시스템입니다.
주어진 텍스트를 분석하여 반드시 파이썬의 dictionary 형식으로 결과를 작성하십시오.
큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다.
아래 dictionary에서 각 value는 지시사항에 해당합니다. 지시사항을 그대로 적지 마시고, 해당 지시사항에 따라 적절한 value를 채워넣으십시오.

분석 결과는 다음 형식으로 작성하십시오:

답변:
{"category": "텍스트의 카테고리를 ['부동산', '산업', '오피니언', '증권'] 중 하나로 분류하여 작성하십시오",
"event_count": "텍스트 내에서 발견된 핵심 이벤트의 개수를 정수로 작성하십시오(증감, 변화, 변동이 나오는 구절을 중점적으로 보세요.)",
"events": ["텍스트에서 추출한 각 핵심 이벤트를 요약하는 문장들을 파이썬 문자열 리스트 형태로 작성하십시오. 원문의 핵심 내용을 그대로 유지하되 한 문장으로 간결하게 작성하십시오"]}"""

# 03_02_02
#   평가셋 326건을 vLLM에 넣을 입력(prompt_lst)과 채점용 정답(label_lst)으로 나눠 준비하는 단계
#   prompt_lst[i]의 정답은 label_lst[i] : 두 목록은 항상 같은 순서로 쌓이므로.

prompt_lst, label_lst = [], []

for _, row in df_eval.iterrows():
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 뉴스 텍스트를 분석해주세요:\n\n{row['text']}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_lst.append(prompt)

    events = [str(row[f"event_sentence{i}"]).strip()
              for i in range(1, 11) if pd.notna(row[f"event_sentence{i}"])]

    # 2. label_lst: 채점용 정답 (326개 dictionary)

    label_lst.append({
        "category": row["text_category"],
        "event_count": int(row["event_quantity"]),
        "events": events,
    })

# 03_02_03

# vLLM이 326건을 한 번에 생성하고, 같은 엔진으로 베이스 모델의 답과 LoRA 모델의 답을 각각 만듬.
# 그 결과를 정답과 비교해 카테고리 정확도, 이벤트 개수 정확도, ROUGE를 계산

sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"])

# llm = LLM(model="Qwen/Qwen3-1.7B", enable_lora=True, max_lora_rank=16, max_model_len=8192)
llm = LLM(model="Qwen/Qwen3-1.7B", enable_lora=True, max_lora_rank=16, max_model_len=8192,
          gpu_memory_utilization=0.75)  

# (1) 베이스: 어댑터 없이
base_preds = [o.outputs[0].text for o in llm.generate(prompt_lst, sampling_params)]

# (2) 파인튜닝: 학습한 LoRA 어댑터를 붙여서
lora_request = LoRARequest("qwen3-ft", 1, "./qwen3-lora-ft-final")
lora_preds = [o.outputs[0].text for o in llm.generate(prompt_lst, sampling_params, lora_request=lora_request)]


# 책의 실행 기록: vLLM이 메모리를 어떻게 나눠 쓰는지와 엔진이 어떤 기능을 키는가?

# # [책 실행 기록: RunPod 80GB GPU, vLLM 0.10.2]
# - 엔진: V1, dtype=bfloat16, prefix caching·chunked prefill(max_num_batched_tokens=8192) 켜짐
# - 어텐션: Flash Attention backend / LoRA 커널: PunicaWrapperGPU (요청마다 다른 어댑터를 한 배치로 처리)
# - 메모리 (gpu_memory_utilization=0.9 → 71.32GiB 확보):
#     가중치 3.25 + 계산용 최대 1.42 + 기타 0.02 → 남은 66.64GiB가 KV 캐시 = 623,888토큰
#     토큰당 KV = 28층 × KV헤드 8 × 128차원 × (K,V) 2 × 2바이트 = 114,688B (약 112KB)
#     8,192토큰 요청을 76.16개 동시에 처리 가능 (623,888 ÷ 8,192)
# - 시작 시간: torch.compile 39.6초 + CUDA graph 캡처 25초(0.6GiB) → 엔진 준비 95.5초
# - 경고: `torch_dtype is deprecated` (평가는 transformers 4.56 이상으로 돌렸다는 흔적)
#         `FlashInfer is not available` (샘플링을 PyTorch로 대체, temperature=0이라 결과는 같음)

# KV 캐시 계산: vLLM 메모리의 핵심.
# 확보한 메모리에서 가중치와 계산용 메모리를 빼고 남는 만큼이 KV 캐시가 되고, 그 크기가 동시에 처리할 수 있는 요청 수를 정합니다. 토큰당 크기를 모델 구조로 직접 계산해 맞춰 볼 수 있음.

# prefix caching: 이번 평가에 특히 효과가 큼. 326건 모두 같은 긴 시스템 프롬프트로 시작해서,
# 그 부분의 KV를 한 번만 계산하고 재사용함.

# 나중에 직접 실행할 때 비교 기준.

# 8GB에서 gpu_memory_utilization=0.75로 돌리는 경우:
# 추정상 KV 캐시는 약 1.25GiB, 1만 1천여 토큰이고 동시 처리는 약 1.4개.

#  - Available KV cache memory
#  - Maximum concurrency
#  - Free memory on device

# 항목	책 (80GB)	내 실행 (8GB, 0.75)
# 확보한 메모리	71.32GiB	5.97GiB
# 가중치 / 계산용	3.25 / 1.42GiB	3.25 / 1.42GiB (같음)
# CUDA graph	0.60GiB	0.38GiB
# KV 캐시	66.64GiB = 623,888토큰	1.29GiB = 12,080토큰
# 동시 처리 (8,192토큰 요청 기준)	76.16개	1.47개
# 엔진 준비 시간	95.5초	41.2초

# 생성: 베이스 6분 21초 vs LoRA 1분 35초 (약 4배 차이)



# 03_02_04

mecab = MeCab()
rouge = Rouge()

def parse_dict(text):
    fragment = text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(fragment)
    except Exception:
        try:
            return ast.literal_eval(fragment)
        except Exception:
            return None

def rouge_f(ref_events, hyp_events):
    ref = " ".join(mecab.morphs(" ".join(ref_events)))
    hyp = " ".join(mecab.morphs(" ".join(hyp_events)))
    if not ref.strip() or not hyp.strip():
        return 0.0, 0.0, 0.0
    s = rouge.get_scores(hyp, ref)[0]      # (예측, 정답) 순서
    return s["rouge-1"]["f"], s["rouge-2"]["f"], s["rouge-l"]["f"]


# 03_02_05: 채점 도구를 준비하는 단계

# 모델의 출력 문자열을 dictionary로 읽어 들이는 함수,
# 이벤트 문장이 정답과 얼마나 비슷한지 ROUGE 점수로 재는 함수를 만듭니다.

def evaluate(preds, labels):
    rows = []
    for pred, label in zip(preds, labels):

        # 출력 문자열 → dictionary
        p = parse_dict(pred)

        # 이벤트 문장을 하나씩 짝지어 비교하지 않고, 전체를 이어 붙여 한 번에 비교합니다.
        # 문장 순서가 다르거나 개수가 다르면 점수가 내려갑니다.
        # 특히 개수가 다르면 재현율이나 정밀도 쪽이 떨어집니다.

        # rouge_f(정답 이벤트, 예측 이벤트): 이벤트 문장 유사도

        # ROUGE-1	형태소 하나 단위로 겹치는 정도 (내용을 얼마나 포함했는지)
        # ROUGE-2	연속된 형태소 두 개 단위로 겹치는 정도 (표현까지 비슷한지)
        # ROUGE-L	가장 긴 공통 부분 순서 (어순까지 맞는지)

        if p is None:
            rows.append({"parse_ok": 0.0, "category_acc": 0.0, "count_acc": 0.0,
                         "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            continue
        pev = p.get("events") if isinstance(p.get("events"), list) else []
        r1, r2, rl = rouge_f(label["events"], pev)

        rows.append({
            "parse_ok":     1.0,
            "category_acc": 1.0 if p.get("category")    == label["category"]    else 0.0,
            "count_acc":    1.0 if p.get("event_count") == label["event_count"] else 0.0,
            "rouge1": r1, "rouge2": r2, "rougeL": rl,
        })

    return pd.DataFrame(rows)

base_df = evaluate(base_preds, label_lst)
lora_df = evaluate(lora_preds, label_lst)

comp = pd.DataFrame({"기존(베이스)": base_df.mean(), "신규(파인튜닝)": lora_df.mean()})
comp["차이"] = comp["신규(파인튜닝)"] - comp["기존(베이스)"]

base_df.to_csv("base_eval.csv", index=False, encoding="utf-8-sig")
lora_df.to_csv("lora_eval.csv", index=False, encoding="utf-8-sig")
comp.to_csv("comparison.csv", encoding="utf-8-sig")

print(comp.round(4))

# vLLM 엔진을 직접 종료 (스크립트 종료 때 나오는 "died unexpectedly" 로그 방지)
llm.llm_engine.engine_core.shutdown()