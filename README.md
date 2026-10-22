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

## 실행 환경 기준 | Runtime Environment

책의 예제는 2장은 Google Colab, 3장 이후는 RunPod 클라우드 GPU를 기준으로 환경을 설정합니다. 이 저장소는 클라우드 GPU를 쓰지 않고, **모든 예제를 로컬 NVIDIA GPU(CUDA)에서 실행하는 것**을 기준으로 합니다.

The book sets up Chapter 2 on Google Colab and Chapter 3 onward on RunPod cloud GPUs. This repository uses no cloud GPUs: every example runs on a local NVIDIA GPU with CUDA.

| 구분 | 책 기준 · Book | 이 저장소 · This repository |
| --- | --- | --- |
| 2장 · Chapter 2 | Google Colab (T4 GPU) | 로컬 CUDA GPU · Local CUDA GPU |
| 3장 이후 · Chapter 3+ | RunPod 클라우드 GPU · RunPod cloud GPU | 로컬 CUDA GPU · Local CUDA GPU |
| GPU | Colab T4 (16GB) 등 · and others | NVIDIA GeForce RTX 5060 Laptop GPU (8GB, Blackwell sm_120) |
| 패키지 설치 · Install | 노트북 셀의 `!pip install` | `uv sync` ([pyproject.toml](pyproject.toml), [uv.lock](uv.lock)) |

GPU 연산이 없는 코드(02_01·02_02의 순수 Python 구현, 02_03의 토크나이저)는 그대로 실행하고, 모델을 GPU에 올리는 코드는 02_04부터입니다. 책의 예제가 8GB보다 큰 GPU 메모리를 전제로 하면, 로컬에 맞춰 조정한 내용을 이 README와 해당 코드에 기록합니다.

Code without GPU work (pure Python in 02_01 and 02_02, the tokenizer in 02_03) runs as is; loading models onto the GPU starts at 02_04. If an example assumes more than 8GB of GPU memory, the local adjustments are recorded here and in the code.

## 코드 변경 원칙 | Code Conventions

- 책 코드를 바꾼 줄은 원래 코드를 주석으로 남기고, 바로 아래에 바꾼 코드를 적습니다.
- GPU를 쓰는 파일은 `_00` 섹션(예: `02_04_00`)에서 모델을 불러오기 전에 GPU를 점검합니다. CUDA 빌드와 사용 가능 여부, GPU 이름과 아키텍처, VRAM, 라이브러리 버전을 출력하고, CUDA를 쓸 수 없으면 `assert`로 멈춥니다.
- 모델을 불러온 뒤에는 모든 파라미터가 GPU에 있는지와 양자화가 적용됐는지 확인합니다. 추론에서는 생성 속도와 최대 GPU 메모리를 출력합니다.
- 실행 결과는 출력 코드 아래에 주석으로 남깁니다. 실행 환경이나 변경 이유처럼 README에 정리한 내용은 코드 주석에 반복하지 않습니다.

Changed lines from the book keep the original as a comment directly above the new line. Files that use the GPU start with a `_00` section that checks the GPU before loading a model, and they verify device placement, quantization, generation speed, and peak GPU memory. Anything documented in this README is not repeated in code comments.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    device_map=device,
    quantization_config=bnb_config
)
```

### `device_map="auto"` → `device_map="cuda:0"`

책은 `device_map="auto"`를 사용합니다. 이 방식은 남은 메모리를 보고 모델을 놓을 장치를 스스로 정합니다. GPU 한 장을 쓰는 로컬에서는 `_00` 섹션에서 정한 `device = "cuda:0"`으로 장치를 직접 지정합니다. 그러면 모델이 GPU에 올라가지 못할 때 바로 알 수 있습니다.

The book uses `device_map="auto"`, which picks devices from available memory. Locally, with a single GPU, the device is set explicitly to `cuda:0` so that a failure to load onto the GPU surfaces immediately.

| 상황 · Situation | `"auto"` | `"cuda:0"` |
| --- | --- | --- |
| GPU를 찾지 못함 (CPU용 PyTorch 등) · No GPU found | 오류 없이 모델 전체를 CPU에 올려 매우 느려짐 · Silently loads on CPU | 바로 오류 · Fails immediately |
| GPU 메모리 부족 · Not enough VRAM | 일부 층을 CPU에 두려다 4비트 모델이라 `ValueError`, 원인 파악이 어려움 · Unclear `ValueError` about CPU offload | 메모리 부족(OOM) 오류로 원인이 분명함 · Clear out-of-memory error |

## 로컬 CUDA 설정 및 실행 | Local CUDA Setup

이 프로젝트는 Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다. 직접 의존성은 [pyproject.toml](pyproject.toml)에, 재현 가능한 전체 의존성 버전은 [uv.lock](uv.lock)에 기록합니다. 소스 파일은 주제별 폴더로 옮기지 않고 저장소 루트에 둡니다.

This project uses Python 3.12 or later and [uv](https://docs.astral.sh/uv/). Direct dependencies are declared in [pyproject.toml](pyproject.toml), with resolved versions recorded in [uv.lock](uv.lock). Source files stay in the repository root rather than being split into topic folders.

| 패키지 | 버전 | 용도 |
| --- | --- | --- |
| `transformers` | 4.55.0 | 토크나이저·모델 로드와 추론 · Tokenizers, model loading, and inference |
| `accelerate` | 1.8.1 | `device_map`으로 모델을 장치에 나눠 올리기 · Device placement for large models |
| `bitsandbytes` | 0.46.0 | 8비트·4비트 양자화 로드 · 8-bit and 4-bit quantized loading |
| `torch` | 2.11.0 (CUDA 12.8) | GPU 텐서 연산과 `return_tensors="pt"` 출력 · GPU tensor backend |

의존성을 추가할 때는 `pip install` 대신 `uv add 패키지==버전`을 사용합니다. 그러면 `pyproject.toml`과 `uv.lock`이 함께 갱신됩니다.

PyTorch는 PyPI가 아니라 [CUDA 12.8용 저장소](https://download.pytorch.org/whl/cu128)에서 받습니다. RTX 50 시리즈(Blackwell)에는 CUDA 12.8 이상이 필요하고, 책의 `bitsandbytes 0.46.0`은 CUDA 12.8까지만 지원합니다. 그래서 두 조건을 모두 만족하는 CUDA 12.8 빌드 중 가장 최신인 torch 2.11.0을 사용합니다. 4비트 양자화는 NVIDIA GPU에서만 동작합니다.

Add dependencies with `uv add package==version` instead of `pip install` so both `pyproject.toml` and `uv.lock` are updated.

PyTorch is installed from the [CUDA 12.8 index](https://download.pytorch.org/whl/cu128) instead of PyPI. RTX 50-series (Blackwell) GPUs need CUDA 12.8 or later, while the book's `bitsandbytes 0.46.0` supports up to CUDA 12.8. torch 2.11.0 is the newest CUDA 12.8 build, so it satisfies both requirements. 4-bit quantization works only on NVIDIA GPUs.

```powershell
# 저장소 폴더에서 실행
uv sync --package llm-serve

# GPU 인식 확인 (True와 GPU 이름이 나와야 함)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 채팅 토크나이저와 템플릿 확인
uv run python .\02_03_llm_chat_template.py

# 4비트 양자화 모델 로드와 GPU 추론
uv run python .\02_04_llm_inference.py
```

**02_03:** 처음 실행할 때 `openai/gpt-oss-20b` 토크나이저 파일을 Hugging Face에서 내려받으므로 인터넷 연결이 필요합니다. 모델 가중치는 내려받지 않습니다. 정상 실행 여부는 Transformers 버전이 `4.55.0`인지, 토큰 ID를 디코딩한 문장이 원문과 같은지, 채팅 템플릿이 어시스턴트 생성 위치까지 출력하는지, 마지막 결과가 PyTorch 텐서인지 확인하면 됩니다.

**02_03:** On first run, the script downloads tokenizer files for `openai/gpt-oss-20b` from Hugging Face, so an internet connection is required. It does not download model weights. To check the setup, confirm Transformers reports version `4.55.0`, decoding the token IDs reproduces the original text, the chat template ends at the assistant generation point, and the final output is a PyTorch tensor.

**02_04:** 처음 실행할 때 `Qwen/Qwen3-4B-Instruct-2507` 가중치(약 8GB)를 내려받습니다. 4비트로 불러오면 GPU 메모리를 약 2.5GB 사용합니다. 모든 파라미터가 `cuda:0`에 있고 `Linear4bit` 층이 252개이면 정상입니다.

**02_04:** On first run, the script downloads the `Qwen/Qwen3-4B-Instruct-2507` weights (about 8GB). Loaded in 4-bit, the model uses about 2.5GB of GPU memory. The setup is correct when every parameter is on `cuda:0` and there are 252 `Linear4bit` layers.

## 저장소 구성 | Repository Contents

코드 파일은 폴더로 재배치하지 않고 저장소 루트에 순서대로 관리합니다.
Code files remain in the repository root and are organized by filename and section ID.

| 파일 | 내용 |
| --- | --- |
| [`01_Theories.md`](01_Theories.md) | 개념 및 학습 메모 · Concepts and notes |
| [`02_01_transformer.py`](02_01_transformer.py) | 자동 미분부터 트랜스포머 인코더-디코더의 학습과 추론까지 · Transformer fundamentals, training, and inference |
| [`02_02_transformer_model_BERT_and_GPT.py`](02_02_transformer_model_BERT_and_GPT.py) | BERT와 GPT 구조 비교 · BERT and GPT architecture |
| [`02_03_llm_chat_template.py`](02_03_llm_chat_template.py) | 토큰화와 채팅 템플릿 적용 · Tokenization and chat templates |
| [`02_04_llm_inference.py`](02_04_llm_inference.py) | GPU 점검, 4비트 양자화 모델 로드와 추론 · GPU checks, 4-bit model loading, and inference |

## 실험 원칙 | How I Learn

- 코드를 직접 실행하고, 입력과 설정을 바꿔 결과를 비교합니다.
- 각 기능이 모델·데이터·서빙 시스템에서 맡는 역할을 기록합니다.
- 재현 가능한 코드와 이해하기 쉬운 설명을 함께 남깁니다.

---

*배우고, 구현하고, 실험하며 LLM 시스템의 전체 그림을 그려 갑니다.*<br>
*Learn, build, and experiment to see the bigger picture of LLM systems.*
