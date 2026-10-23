# 9장 LLM 에이전트 실험 노트

코드: [09_llm_agents.py](09_llm_agents.py) · 실행 로그: `logs/09/` (커밋하지 않음)

이 노트는 9장 실험에서 관찰한 결과를 **GPT-4 이후 주요 논문의 흐름**과 연결해 해석합니다. 책은 `gpt-4o`를 쓰지만, 이 저장소는 쓸 수 있는 모델인 `gpt-6-luna`와 `gpt-5.6-luna`로 실행했습니다. 두 모델 모두 `stop`과 `temperature`를 받지 않습니다(직접 확인).

---

## 흐름 한눈에 보기

| 시기 | 논문·발표 | 핵심 | 이 실험과의 연결 |
| --- | --- | --- | --- |
| 2022.01 | Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | 생각 과정을 글로 쓰게 하면 추론이 좋아짐 | ReAct의 `Thought:` |
| 2022.03 | Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT) | RLHF로 지시 따르기 학습 | 프롬프트 지시가 통하는 바탕 |
| 2022.03 | Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* | 여러 번 샘플링해 다수결 | 결과가 매번 다른 것을 활용하는 관점 |
| 2022.10 | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* | 생각과 행동(도구)을 번갈아 텍스트로 | 09_01 `create_react_agent` |
| **2023.02** | Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* | 모델이 **언제·어떤 도구를 부를지** 스스로 학습, 도움이 되는 호출만 남김 | 불필요한 요약 호출 |
| **2023.03** | OpenAI, *GPT-4 Technical Report* | 지시 따르기와 추론의 큰 도약 | 이 흐름의 기준점 |
| 2023.05 | Patil et al., *Gorilla: Large Language Model Connected with Massive APIs* | API 호출을 정확한 형식으로 내도록 미세조정 | 구조화된 도구 호출 |
| **2023.06** | OpenAI 펑션 콜링 (발표) | 도구 호출을 **JSON(`tool_calls`)**으로 출력 | `stop`이 필요 없어진 계기, 09_02 |
| 2023.07 | Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs* | 대규모 실제 API로 도구 사용 학습 | 도구 사용이 학습 대상이 됨 |
| 2023.10 | Huang et al., *MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use* | 도구를 **쓸지 말지** 판단하는 능력 평가 | 불필요한 요약 호출 |
| 2023.11 | Zhou et al., *Instruction-Following Evaluation for Large Language Models* (IFEval) | 형식 지시를 지키는지 검증 가능한 방식으로 평가 | 프롬프트 STOP 지시 |
| 2024.02 | Wang et al., *Executable Code Actions Elicit Better LLM Agents* (CodeAct) | 행동을 텍스트 대신 실행 가능한 코드로 | 텍스트 ReAct 이후의 방향 |
| **2024.04** | Wallace et al., *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions* | 시스템·개발자 지시를 사용자 입력보다 우선하도록 학습 | 도구 설명의 "반드시"가 판단을 이김 |
| **2024.09** | OpenAI o1 (*Learning to Reason with LLMs*, *o1 System Card*) | 강화학습으로 숨은 추론을 길게 한 뒤 답함, 샘플링 매개변수 제한 | `stop`·`temperature` 미지원 |
| **2025.01** | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | 강화학습만으로 추론 능력을 끌어냄 (공개 모델) | 추론이 모델 안으로 들어감 |

굵게 표시한 것이 GPT-4 이후 이 실험을 이해하는 데 핵심인 지점입니다.

---

## 09_01 ReAct (`gpt-6-luna`, `stop_sequence=False`)

실행: 2026-10-23 17:15, 약 23초, 로그 `logs/09/09_01_20261023_171521.log`

도구 호출 3번(재무 → 뉴스 검색 → 요약) 뒤 최종 답변이 나왔습니다. 형식 오류는 없었고 `max_iterations` 6 안에서 끝났습니다.

### 분석 1. 프롬프트의 STOP 지시가 `stop` 매개변수를 대신했다

**관찰:** 매 턴이 `Action Input: …`에서 정확히 끝났습니다. 이어진 Observation은 모두 실제 도구 결과였습니다. yfinance가 돌려준 가공되지 않은 값 `PER 6.5031686`이 그대로 들어온 것이 증거입니다. 모델이 Observation을 지어낸 흔적은 없습니다.

**흐름 속 의미:**
- **ReAct(2022)의 구현 방식:** 원 논문은 few-shot 프롬프트로 형식을 가르쳤습니다. LangChain 같은 프레임워크는 모델이 `Observation`까지 지어내는 것을 막으려고 **디코딩 단계의 `stop`**으로 출력을 잘랐습니다. 당시 모델은 "여기서 멈춰라" 같은 형식 지시를 믿고 맡기기 어려웠습니다.
- **지시 따르기의 발전:** InstructGPT(2022)의 RLHF에서 GPT-4(2023)로 오면서 지시 따르기가 크게 좋아졌습니다. IFEval(2023)은 "이 형식을 지켜라" 같은 지시를 검증 가능한 방식으로 평가하는 기준을 만들었습니다. 이번 실험에서 프롬프트 한 줄로 멈춤이 지켜진 것은 **멈춤을 제어하는 수단이 디코딩 설정에서 지시 따르기로 옮겨 왔음**을 보여 줍니다.
- **구조적 해결:** 펑션 콜링(2023.06)부터는 도구 호출이 JSON으로 나오고 그 턴이 프로토콜상 끝나므로, 애초에 `stop`이 필요 없습니다. Toolformer·Gorilla·ToolLLM(2023)이 도구 호출 자체를 학습 대상으로 만든 흐름의 결과입니다. o1(2024) 같은 추론 모델과 Responses API에서 `stop`이 빠진 것은 이 흐름의 끝에 있습니다.

### 분석 2. 요약 도구 호출은 불필요했다

**관찰:** 모델은 검색 결과를 이미 스스로 줄여 쓴 뒤, 그 글을 `summarize_text`에 다시 넣었습니다. 요약 모델 호출이 한 번 더 일어나 비용과 시간만 늘었습니다.

**흐름 속 의미:**
- **Toolformer(2023):** 도움이 되는 API 호출만 남기는 기준으로 학습해서, "**필요할 때만** 도구를 부르는 것"을 목표로 삼았습니다.
- **MetaTool(2023):** 도구를 **쓸지 말지** 판단하는 것 자체가 어려운 능력임을 평가로 보여 줬습니다. 모델은 도구를 과하게 쓰거나 덜 쓰는 경향이 있습니다.
- **Instruction Hierarchy(2024):** 이번 호출의 직접 원인은 도구 설명의 "요약할 때는 **반드시** 이 도구를 사용하세요"입니다. 도구 설명은 개발자가 준 지시라서, 최신 모델은 이를 높은 우선순위로 따르도록 학습돼 있습니다. 그 결과 "이미 요약했다"는 모델 자신의 판단보다 개발자 지시가 이겼습니다.
- **실무 교훈:** 도구 설명에는 "반드시"처럼 무조건 지시하는 표현보다 **사용 조건**("검색 결과가 길어서 직접 요약하기 어려울 때 사용")을 쓰는 것이 낫습니다. 에이전트의 행동과 비용은 모델만큼이나 도구 설명 한 줄에 좌우됩니다.

### 분석 3. 실행할 때마다 결과가 달라진다

**관찰:** 09_01_02에서 직접 검색한 결과와, 에이전트가 검색한 결과가 서로 다릅니다. 같은 질문을 다시 실행하면 도구 순서와 문장도 달라질 수 있습니다.

**흐름 속 의미:**
- **모델 쪽 원인 (샘플링):** luna 모델은 `temperature`를 받지 않아 기본값 1로 샘플링합니다. o1(2024) 이후의 추론 모델은 강화학습으로 훈련된 추론 과정을 보호하려고 샘플링 조절을 막는 경향이 있습니다. "책처럼 `temperature=0`으로 매번 같은 결과를 낸다"는 재현 방식을 사용자가 더 이상 직접 고를 수 없게 된 것입니다.
- **다르게 나오는 것을 쓰는 관점:** Self-Consistency(2022)는 여러 번 샘플링해 다수결을 하면 추론이 좋아진다는 것을 보였습니다. 결과가 매번 다르다는 것은 결점이 아니라, **여러 번 실행해 분포로 평가해야 한다**는 뜻이기도 합니다.
- **환경 쪽 원인:** DuckDuckGo 검색 결과는 시간에 따라 바뀝니다. 에이전트는 바깥 세계를 도구로 읽기 때문에, 모델이 같아도 결과는 달라집니다.
- **평가에 주는 교훈:** 에이전트는 실행 한 번으로 판단하지 말고, 여러 번 실행해서 도구 호출 순서·성공률·비용의 분포를 봐야 합니다. 그래서 이 저장소는 실행마다 `logs/09/`에 원본을 남깁니다.

---

## 09_02 펑션 콜링 · 09_03 예외 처리와 병렬 실행 · 09_04 Text-to-SQL

(실행 후 추가)
