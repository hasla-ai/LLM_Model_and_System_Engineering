'''
9장 LLM 에이전트 (책 9.3~9.6, OpenAI API)
  09_01 ReAct               (9.3, chapter09_1_react.ipynb)
  09_02 펑션 콜링            (9.4, chapter09_2_function_calling.ipynb)
  09_03 예외 처리·병렬 실행   (9.5, chapter09_3_exception_parallel.ipynb)
  09_04 Text-to-SQL          (9.6, chapter09_4_text_to_sql.ipynb)

실행: uv run python 09_llm_agents.py 09_02 09_03    # 인자가 없으면 전체 실행
'''

import os, sys, getpass, asyncio


## 09_00 공통 설정 (네 노트북에 반복되던 설치·키 셀을 한 번으로)

# !pip install langchain_openai==1.1.12 langchain_community==0.4.1 langchain==1.2.14 sqlalchemy numexpr pydantic tenacity nest_asyncio
# !pip install -U duckduckgo_search==7.5.1 yfinance ddgs
# → pyproject.toml로 대체

# 셸에서 read -s로 넣은 키가 있으면 그대로 쓰고, 없을 때만 입력받는다
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# 노트북의 이벤트 루프 때문에 넣은 것이라 .py에서는 필요 없다 (비동기는 asyncio.run 사용)
# import nest_asyncio
# nest_asyncio.apply()


def part_09_01():
    '''
    시가총액, PER 조회 및 뉴스 검색/도구/요약하는 추론-행동 ReAct Agent
    ReAct 프롬프트 (Thought → Action → Observation 반복 형식)

    검색 도구: DuckDuckGoSearchRun()
    시가총액, PER 조회 등 금융 데이터 API : Yahoo Finance
    '''

    ## 09_01_02

    from langchain_community.tools import DuckDuckGoSearchRun

    search = DuckDuckGoSearchRun()
    print(search.invoke("현대자동차 최근 뉴스"))

    ## 09_01_03

    import yfinance as yf
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_classic.tools import Tool
    from langchain_core.prompts import PromptTemplate

    ## 09_01_04
    # 1. 실제 금융 데이터 API 도구 (Yahoo Finance)

    def get_financial_statement(symbol: str) -> str:
        """특정 기업의 티커를 입력받아 시가총액과 PER 등 재무 정보를 반환합니다."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap', '정보 없음')
            forward_pe = info.get('forwardPE', '정보 없음')
            # 시가총액을 조 단위로 변환 (숫자일 경우)
            if isinstance(market_cap, (int, float)):
                market_cap = f"{market_cap / 1_000_000_000_000:.2f}조 원"  ## 한국 종목

            return f"{symbol}의 시가총액은 {market_cap}이며, 추정 PER은 {forward_pe}입니다."
        except Exception as e:
            return f"재무 정보를 불러오는 데 실패했습니다: {e}"

    ## 09_01_05
    # 2. 실시간 뉴스 검색 도구 (DuckDuckGo)

    search_tool = DuckDuckGoSearchRun()

    # 3. 텍스트 요약 도구 (LLM을 도구 안에서 다시 호출)
    def summarize_text(text: str) -> str:
        """검색된 긴 텍스트를 입력받아 핵심만 3줄로 요약합니다."""
        summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = summary_llm.invoke(f"다음 뉴스 텍스트를 3문장 이내로 핵심만 요약해:\n{text}")
        return response.content

    ## 09_01_06

    # 4. 도구 리스트 및 정교한 설명(Description) 설정
    tools = [
        Tool(
            name="get_financial_statement",
            func=get_financial_statement,
            description="기업의 재무 정보(시가총액, PER 등)를 조회합니다. 한국 주식은 종목코드 뒤에 '.KS'를 붙이세요 (예: 현대차 -> 005380.KS)."
        ),
        Tool(
            name="get_latest_news",
            func=search_tool.run,
            description="기업 관련 최신 뉴스를 검색합니다."
        ),
        Tool(
            name="summarize_text",
            func=summarize_text,
            description="매우 긴 뉴스나 검색 결과를 요약할 때 사용합니다. 요약할 때는 반드시 이 도구를 사용하세요."
        )
    ]

    ## 09_01_07

    # 5. ReAct 표준 프롬프트 (문자열 안의 줄은 들여쓰지 않는다: 들여쓰면 프롬프트에 공백이 들어감)
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer in KOREAN to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    react_prompt = PromptTemplate.from_template(template)

    ## 09_01_08

    # 6. 모델 및 에이전트 실행기 구성

    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna", temperature=0)

    agent = create_react_agent(llm, tools, react_prompt)


    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True)

    # 7. 실행
    user_query = "한국 주식 현대차(005380.KS)의 실제 재무 정보와, 최근 현대자동차 관련 뉴스를 검색해서 요약해줘."
    response = agent_executor.invoke({"input": user_query})
    print("\n[최종 답변]:\n", response["output"])


def part_09_02():
    ## 09_02_01 함수 스키마 되짚기 (작성 예정)
    print("09_02: 아직 작성 전")


def part_09_03():
    ## 09_03_01 도구 호출 거부와 예외 처리 (작성 예정)
    # 노트북의 최상위 await는 .py에서 쓸 수 없으므로 asyncio.run(...)으로 바꾼다
    print("09_03: 아직 작성 전")


def part_09_04():
    ## 09_04_01 Text-to-SQL (작성 예정, hospital.db는 실행할 때 새로 만들어지므로 .gitignore)
    print("09_04: 아직 작성 전")


PARTS = {"09_01": part_09_01, "09_02": part_09_02, "09_03": part_09_03, "09_04": part_09_04}

if __name__ == "__main__":
    for name in (sys.argv[1:] or PARTS):
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        PARTS[name]()
