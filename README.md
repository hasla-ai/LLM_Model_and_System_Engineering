# LLM_Model_and_System_Engineering

> **모델을 이해하고, 시스템을 직접 만들며 배우는 LLM 엔지니어링 실험실**<br>
> *A hands-on lab for understanding LLMs and building the systems around them.*

![A map of the topics explored in this LLM engineering project](llm-engineering-map.svg)

## 프로젝트 소개 | About

### 한국어

저는 아직 뛰어난 실력자라고 말하기 어렵지만, 한국어로 된 *LLM 마스터: 모델×시스템 엔지니어링*이 출간되었다는 사실이 무척 반갑습니다. 트랜스포머의 기초에서 출발해 SFT·VLM·툴콜링 파인튜닝, vLLM 서빙, 강화학습, 고급 RAG와 에이전트까지 최신 LLM 기술을 차근차근 쌓아 가는 구성이 특히 유익하게 다가왔습니다.

이 저장소는 책과 공개 예제 코드를 길잡이 삼아, 각종 예제를 직접 실행하고 변형하면서 기능을 구현해 보는 학습 기록입니다. 코드를 따라 쓰는 데서 그치지 않고, 각 구성 요소가 어떻게 연결되는지 실험하며 LLM 서빙 시스템의 전반적인 흐름을 누구나 이해하기 쉽게 정리해 가는 것을 목표로 합니다.

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

## 저장소 구성 | Repository Contents

- [`02_01_transformer.py`](02_01_transformer.py) — 트랜스포머 관련 실습 코드 | Transformer exercise
- [`01_Theories.md`](01_Theories.md) — 학습 메모 | Study notes

## 실험 원칙 | How I Learn

- 예제를 직접 실행하고, 입력과 설정을 바꿔 결과를 비교합니다.
- 각 기능이 모델·데이터·서빙 시스템에서 맡는 역할을 기록합니다.
- 재현 가능한 코드와 이해하기 쉬운 설명을 함께 남깁니다.

---

*배우고, 구현하고, 실험하며 LLM 시스템의 전체 그림을 그려 갑니다.*<br>
*Learn, build, and experiment to see the bigger picture of LLM systems.*
