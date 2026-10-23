# 9장 에이전트 실험 요약

모델 `gpt-6-luna` · `gpt-5.6-luna` (둘 다 `stop`·`temperature` 미지원) · LangChain 1.2.14 (`langchain-classic`) · 상세: [09_llm_agents.md](09_llm_agents.md)

| 파트 | 결과 | 핵심 관찰 | 관련 흐름 |
| --- | --- | --- | --- |
| 09_01 ReAct | 도구 3번 → 답, 23초, 형식 오류 0 | ① 프롬프트 STOP 지시가 `stop`을 대신함 ② 요약 도구를 불필요하게 호출 ③ 실행마다 결과 다름 | ① InstructGPT → GPT-4 → IFEval, 펑션 콜링·o1 ② Toolformer · MetaTool · Instruction Hierarchy ③ Self-Consistency · o1 |
| 09_02 펑션 콜링 | (실행 후) | | |
| 09_03 예외 처리·병렬 | (실행 후) | | |
| 09_04 Text-to-SQL | (실행 후) | | |

**결론:** 텍스트 ReAct는 최신 추론 모델에서도 프롬프트 지시만으로 동작합니다. 다만 멈춤, 도구 사용 판단, 재현성을 제어하는 수단이 디코딩 설정(`stop`, `temperature`)에서 모델 학습(지시 따르기, 도구 사용, 추론 강화학습)으로 넘어갔습니다.
