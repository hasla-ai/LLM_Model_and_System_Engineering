# LLM_Model_and_System_Engineering

> **모델을 이해하고, 시스템을 직접 만들며 배우는 LLM 엔지니어링 실험실**<br>
> *A hands-on lab for understanding LLMs and building the systems around them.*

![A map of the topics explored in this LLM engineering project](llm-engineering-map.svg)

## 프로젝트 소개 | About

### 한국어

저는 아직 뛰어난 실력자라고 말하기 어렵지만, 한국어로 된 *LLM 마스터: 모델×시스템 엔지니어링*이 출간되었다는 사실이 무척 반갑습니다. 트랜스포머의 기초에서 출발해 SFT·VLM·툴콜링 파인튜닝, vLLM 서빙, 강화학습, 고급 RAG와 에이전트까지 최신 LLM 기술을 차근차근 쌓아 가는 구성이 특히 유익하게 다가왔습니다.

이 저장소는 책과 공개 코드를 길잡이 삼아, 코드를 직접 실행하고 변형하면서 기능을 구현해 보는 학습 기록입니다. 각 구성 요소가 어떻게 연결되는지 살펴보고, LLM 서빙 시스템의 전반적인 흐름을 이해하기 쉽게 정리해 가는 것을 목표로 합니다.

### English

I am still learning and would not call myself an expert, so I am especially glad to see a Korean book dedicated to LLM model and systems engineering. Its gradual path—from transformer fundamentals through SFT, VLMs, tool-calling fine-tuning, vLLM serving, reinforcement learning, advanced RAG, and agents—offers a helpful way to build up knowledge of current LLM technologies.

This repository is my learning log as I work through the book and its public code examples. I run and adapt the examples, implement features, and experiment with how the pieces fit together. The goal is to make the overall shape of LLM serving easier to understand through hands-on practice.

## 참고 도서 및 코드 | Book & Code

- **도서 | Book:** *LLM 마스터: 모델×시스템 엔지니어링*
- **저자 | Authors:** 에디 유, 양기빈, 장지선, 조수현
- **출판사 | Publisher:** 프리렉
- **출간일 | Published:** 2026-09-28

<p align="center">
  <img src="assets/llm-master-cover.png" alt="LLM 마스터: 모델×시스템 엔지니어링 도서 이미지" width="420">
</p>

- **공개 예제 코드 | Public example code:** [freelec-llm-scientist/freelec_llm_scientist_code](https://github.com/freelec-llm-scientist/freelec_llm_scientist_code)
- **함께 살펴보는 노트북 | Notebook reference:** [Chapter 2: Transformer](https://github.com/freelec-llm-scientist/freelec_llm_scientist_code/blob/main/chapter_02/chapter02_1_transformer.ipynb)

이 프로젝트는 학습과 실험을 위한 개인 저장소이며, 도서의 저자·출판사 또는 원본 코드 저장소와 공식적으로 제휴하거나 이를 대표하지 않습니다.<br>
This is an independent learning project and is not affiliated with or endorsed by the book's authors, publisher, or the source-code repository.

## 학습 로드맵 | Learning Roadmap

| 주제 | Topics |
| --- | --- |
| 트랜스포머 | Transformer fundamentals |
| 파인튜닝 | Fine-tuning: SFT, VLM, and tool calling |
| 모델 서빙 | Model serving with vLLM |
| 추론과 강화학습 | Reasoning and reinforcement learning |
| 검색 증강 생성 | Advanced retrieval-augmented generation (RAG) |
| 에이전트 | LLM agents |

## 환경 설정 및 실행 | Environment Setup

이 프로젝트는 Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다. 직접 의존성은 [pyproject.toml](pyproject.toml)에, 재현 가능한 전체 의존성 버전은 [uv.lock](uv.lock)에 기록합니다. 소스 파일은 주제별 폴더로 옮기지 않고 저장소 루트에 둡니다.

This project uses Python 3.12 or later and [uv](https://docs.astral.sh/uv/). Direct dependencies are declared in [pyproject.toml](pyproject.toml), with resolved versions recorded in [uv.lock](uv.lock). Source files stay in the repository root rather than being split into topic folders.

```powershell
# 저장소 폴더에서 실행
uv sync --package llm-serve

# 채팅 토크나이저와 템플릿 확인
uv run python .\02_03_llm_chat_template.py
```

처음 실행할 때 `openai/gpt-oss-20b` 토크나이저 파일을 Hugging Face에서 내려받으므로 인터넷 연결이 필요합니다. 모델 가중치는 내려받지 않습니다. `return_tensors="pt"` 출력을 위해 PyTorch도 의존성에 포함되어 있습니다.

On first run, the script downloads tokenizer files for `openai/gpt-oss-20b` from Hugging Face, so an internet connection is required. It does not download model weights. PyTorch is included for the `return_tensors="pt"` output.

정상 실행 여부는 Transformers 버전이 `4.55.0`인지, 토큰 ID를 디코딩한 문장이 원문과 같은지, 채팅 템플릿이 어시스턴트 생성 위치까지 출력하는지, 마지막 결과가 PyTorch 텐서인지 확인하면 됩니다.

To check the setup, confirm Transformers reports version `4.55.0`, decoding the token IDs reproduces the original text, the chat template ends at the assistant generation point, and the final output is a PyTorch tensor.

## 저장소 구성 | Repository Contents

코드 파일은 폴더로 재배치하지 않고 저장소 루트에 순서대로 관리합니다.
Code files remain in the repository root and are organized by filename and section ID.

| 파일 | 내용 |
| --- | --- |
| [`01_Theories.md`](01_Theories.md) | 개념 및 학습 메모 · Concepts and notes |
| [`02_01_transformer.py`](02_01_transformer.py) | 자동 미분부터 트랜스포머 인코더-디코더의 학습과 추론까지 · Transformer fundamentals, training, and inference |
| [`02_02_transformer_model_BERT_and_GPT.py`](02_02_transformer_model_BERT_and_GPT.py) | BERT와 GPT 구조 비교 · BERT and GPT architecture |
| [`02_03_llm_chat_template.py`](02_03_llm_chat_template.py) | 토큰화와 채팅 템플릿 적용 · Tokenization and chat templates |

## 실험 원칙 | How I Learn

- 코드를 직접 실행하고, 입력과 설정을 바꿔 결과를 비교합니다.
- 각 기능이 모델·데이터·서빙 시스템에서 맡는 역할을 기록합니다.
- 재현 가능한 코드와 이해하기 쉬운 설명을 함께 남깁니다.

---

*배우고, 구현하고, 실험하며 LLM 시스템의 전체 그림을 그려 갑니다.*<br>
*Learn, build, and experiment to see the bigger picture of LLM systems.*
