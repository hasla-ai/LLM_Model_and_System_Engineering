# 1. 라이브러리 불러오기


from transformers import AutoTokenizer

# 2. 모델 토크나이저 불러오기


# 예시: openai/gpt-oss-20b 모델의 토크나이저 사용
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("모델에 사용되는 토크나이저:", tokenizer.__class__.__name__)
print("='-'" * 30)

# 모델에 사용되는 토크나이저: PreTrainedTokenizerFast
# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'


# 3. 일반 텍스트 토큰화 예시


text = "안녕하세요. LLM은 정말 흥미롭습니다."
print(f"원본 텍스트: '{text}'")

# text를 토큰들의 리스트로 변환
tokens = tokenizer.tokenize(text)
print(f"토큰 리스트: {tokens}")

# 토큰 리스트를 고유한 ID로 변환
token_ids = tokenizer.encode(text)
print(f"토큰 ID 리스트: {token_ids}")

# 토큰 ID를 다시 텍스트로 변환
decoded_text = tokenizer.decode(token_ids)
print(f"복원된 텍스트: '{decoded_text}'")
print("='-'" * 30)

# 원본 텍스트: '안녕하세요. LLM은 정말 흥미롭습니다.'
# 토큰 리스트: ['ìķĪ', 'ëħķíķĺìĦ¸ìļĶ', '.', 'ĠL', 'LM', 'ìĿĢ', 'Ġìłķë§Ĳ', 'ĠíĿ', '¥', 'ë¯¸', 'ë¡Ń', 'ìĬµëĭĪëĭ¤', '.']
# 토큰 ID 리스트: [14307, 171731, 13, 451, 19641, 4740, 110308, 48290, 98, 14561, 101923, 12020, 13]
# 복원된 텍스트: '안녕하세요. LLM은 정말 흥미롭습니다.'
# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'



# 4. 챗 템플릿 적용 예시

# 대화 기록을 역할(role)과 내용(content)으로 구성된 리스트로 정의
chat_history = [
    {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
    {"role": "user", "content": "LLM은 무엇인가요?"},
    {"role": "assistant", "content": "LLM은 Large Language Model의 약자로, 대규모 언어 모델을 뜻합니다."},
    {"role": "user", "content": "어떻게 작동하나요?"}
]

# 챗 템플릿을 적용하여 모델 입력 문자열 생성
# `tokenize=False`로 설정하면 토큰화되지 않은 문자열을 반환
# `add_generation_prompt=True`로 설정하면 어시스턴트의 답변을 생성하기 위한 프롬프트 추가
formatted_chat_string = tokenizer.apply_chat_template(
    chat_history,
    tokenize=False,
    add_generation_prompt=True
)

print("챗 템플릿이 적용된 입력 문자열:")
print(formatted_chat_string)
print("='-'" * 30)

# 챗 템플릿이 적용된 입력 문자열:
# <|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
# Knowledge cutoff: 2024-06
# Current date: 2026-10-22

# Reasoning: medium

# # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

# 당신은 유용한 AI 어시스턴트입니다.

# <|end|><|start|>user<|message|>LLM은 무엇인가요?<|end|><|start|>assistant<|channel|>final<|message|>LLM은 Large Language Model의 약자로, 대규모 언어 모델을 뜻합니다.<|end|><|start|>user<|message|>어떻게 작동하나요?<|end|><|start|>assistant
# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'



# 5. 챗 템플릿이 적용된 문자열을 토큰 ID로 변환

formatted_chat_ids = tokenizer.apply_chat_template(
    chat_history,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

print("챗 템플릿이 적용된 토큰 ID 리스트 (Tensor):")
print(formatted_chat_ids)

# ='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'='-'
# 챗 템플릿이 적용된 토큰 ID 리스트 (Tensor):
# tensor([[200006,  17360, 200008,   3575,    553,  17554, 162016,     11,    261,
#            4410,   6439,   2359,  22203,    656,   7788,  17527,    558,  87447,
#          100594,     25,    220,   1323,     19,     12,   3218,    198,   6576,
#            3521,     25,    220,   1323,     21,     12,    702,     12,   1709,
#             279,  30377,    289,     25,  14093,    279,      2,  13888,  18403,
#              25,   8450,     11,  49159,     11,   1721,     13,  21030,   2804,
#             413,   7360,    395,   1753,   3176,     13, 200007, 200006,  77944,
#          200008,      2,  68406,    279,  19388,  11753,   4740,  22748,   8423,
#            3748,  20837,  21252,   5637,   5648,  74404,   7984,  27001,    364,
#          200007, 200006,   1428, 200008,   7454,     44,   4740, 103740, 149298,
#            7952,     30, 200007, 200006, 173781, 200005,  17196, 200008,   7454,
#              44,   4740,  27976,  20333,   8186,   3408,  58496, 162282,     11,
#           10985,  52580,  17267,  69163,   5959, 128372,   3281, 152898,  24490,
#              13, 200007, 200006,   1428, 200008,   5959,  33854,  91205,  31389,
#           11803,   3131, 151413,     30, 200007, 200006, 173781]])