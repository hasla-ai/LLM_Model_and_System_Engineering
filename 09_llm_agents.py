'''
9장 LLM 에이전트 (OpenAI API)

  09_01 ReAct               
  09_02 펑션 콜링           
  09_03 예외 처리·병렬 실행  
  09_04 Text-to-SQL         

  개별 실행

  기록 (실행 결과는 각 섹션 아래 [실행 결과] 주석에도 핵심만 남김)
    - 원본 로그: logs/09/<파트>_<시각>.log  (실행할 때 자동 저장, 커밋하지 않음)
    - 상세 분석: 09_llm_agents.md          (관찰을 GPT-4 이후 주요 논문 흐름과 연결한 긴 노트)
    - 압축 요약: 09_agents_analysis.md     (파트별 결과와 핵심 관찰을 한 장으로)
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

    (실행) Ubuntu, 한 줄 씩 실행
     read -rs -p "OpenAI API Key: " OPENAI_API_KEY && export OPENAI_API_KEY && echo   # 이 줄만 실행하고 키를 붙여 넣기

    echo ${#OPENAI_API_KEY}     # 키 값은 보이지 않고 길이만 출력. 0이거나 몇 글자뿐이면 붙여 넣기가 안 된 것

    curl -s -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
    
    # 200: 키가 정상.
    # 401: 키가 삭제·폐기됐거나 잘못 복사된 것.
    # 429 insufficient_quota: 키는 정상인데 크레딧이 없거나 결제 설정 잘못된 경우.

    uv run python 09_llm_agents.py 09_01
    '''

    ## 09_01_02

    from langchain_community.tools import DuckDuckGoSearchRun

    search = DuckDuckGoSearchRun()
    print(search.invoke("현대자동차 최근 뉴스"))

# ==================== 09_01 ====================
# 현대자동차그룹의 최신 뉴스와 공식 보도자료를 가장 빠르게 전달합니다. 지속가능한 미래를 위한 기술 혁신과 주요 소식을 확인하세요. Hyundai의 최신 뉴스와 차량 관련 업데이트를 한곳에서 확인하세요. 신차 출시, 기술 동향, 브랜드 소식을 빠르고 정확하게 전달합니다. 현대차가 6년 만에 선보이는 5세대 완전변경 모델 '디 올 뉴 투싼'을 공개했다. 디자인 언어 '아트 오브 스틸'을 바탕으로 강인한 외관을 완성했으며 전장·전폭·휠베이스를 확대해 동급 최고 수준의 공간성을 확보했다. 파워트레인은 1.6 터보 가솔린 (최고 출력 193PS)과 차세대 하이브리드 시스템 (시스템 최고 출력 245PS·복합연비 17.4km/ℓ) 두 가지로 운영된다. 현대차 SUV 최초로 플레오스 커넥트를 탑재해 생성형 AI 에이전트 '글레오 AI'와 '플레오스 앱마켓' 기반의 디지털 경험을 제공한다. Hyundai 제품 최신 뉴스와 소유자에게 실용적인 공지사항을 안내하는 페이지입니다. 필요한 정보를 쉽게 찾아보세요. 현대자동차가 인도네시아 발리에 전기버스를 공급하며 아시아·태평양 지역 친환경 상용차 시장 공략에 속도를 낸다. 현대차는 22일 (현지시간) 인도네시아 발리 독립기념탑에서 열린 '전기버스 및 충전 인프라 보급 프로젝트 출범식'에서 카운티 일렉트릭 전기버스 11대를 발리 주정부에 공급했다고 23일 밝혔다. 이날 행사에는 김영신 현대차...

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
        # summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary_llm = ChatOpenAI(model="gpt-5.6-luna")
        # 쓸 수 있는 가벼운 모델, temperature 미지원 → 기본값(1)

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
    # (추가) "IMPORTANT: …" 한 줄:
    # stop 매개변수를 못 쓰는 모델이라, Action Input 뒤에서 멈추라고 프롬프트로 지시

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

IMPORTANT: After writing "Action Input: ...", STOP immediately. Never write the Observation yourself.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    react_prompt = PromptTemplate.from_template(template)


    ## 09_01_08

    # 6. 모델 및 에이전트 실행기 구성

    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna")   # temperature 미지원 → 기본값(1), 실행할 때마다 답이 조금씩 달라짐

    # agent = create_react_agent(llm, tools, react_prompt)

    agent = create_react_agent(llm, tools, react_prompt, stop_sequence=False)   # stop 미지원 모델: 자동 stop 끄기


    # ReAct는 LLM이 Action: …까지만 쓰고 멈춰야 합니다.
    # 그래야 코드가 도구를 실행해 진짜 결과를 Observation에 넣어 줍니다.

    # 멈추지 않으면 LLM이 Observation: …까지 스스로 지어내서(환각) 도구를 실행하지 않은 채 답을 꾸밀 수 있습니다.
    # 그런데 gpt-6-luna는 이 stop 매개변수를 받지 않는 모델입니다. 추론형 모델 중에 이런 모델이 있습니다.


    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
    #                                handle_parsing_errors=True)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True,
                                   max_iterations=6)   # 형식이 깨져 반복해도 API 호출은 6번에서 멈춤 (기본 15)

    # 7. 실행
    user_query = "한국 주식 현대차(005380.KS)의 실제 재무 정보와, 최근 현대자동차 관련 뉴스를 검색해서 요약해줘."
    response = agent_executor.invoke({"input": user_query})
    print("\n[최종 답변]:\n", response["output"])

    # [로그 저장] logs/09/09_01_20261023_171521.log

    # [실행 결과] 2026-10-23 17:15, gpt-6-luna(요약 도구 gpt-5.6-luna), stop_sequence=False, 약 23초
    # 도구 호출 3번 → Final Answer (형식 오류 없음, max_iterations 6 안에서 끝남, LLM 호출 4번 + 요약 1번)
    #   ① get_financial_statement("005380.KS") → 시가총액 77.37조 원, 추정 PER 6.5031686 (yfinance 실제 값)
    #   ② get_latest_news("현대자동차 최근 뉴스") → '디 올 뉴 투싼' 공개, 발리 전기버스 11대 공급 등
    #   ③ summarize_text(모델이 먼저 줄여 쓴 글) → 3문장 요약
    # 최종 답변: 시가총액 약 77.37조 원, PER 약 6.50배 ("기준일이 없으니 참고 지표로 보라"는 단서 포함) + 뉴스 2건 요약
    # 분석(프롬프트 STOP 지시, 불필요한 요약 호출, 결과가 매번 다른 이유): 09_llm_agents.md 참고


def part_09_02():
    '''
    펑션 콜링: LLM이 도구 호출을 JSON(함수 이름 + 인자)으로 제안하고, 실제 실행은 코드가 한다
    09_02_01 함수 스키마 / 09_02_02 bind_tools / 09_02_03 FC + ReAct (멀티턴 메모리) / 09_02_04 병렬 도구 제안
    '''

    ## 09_02_01 함수 스키마.
    # LLM에게 전달되는 도구 설명서의 실제 모양

    weather_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 지역의 현재 날씨를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "조회할 지역 이름"}
                },
                "required": ["location"]
            }
        }
    }
    # 노트북은 셀의 마지막 값을 화면에 보여 주지만 .py는 그렇지 않으므로 변수에 담아 출력
    print(weather_schema)

    ## 09_02_02 bind_tools

    # 날씨 도구(Open-Meteo)를 붙이면 LLM이 {'location': 'Seoul'}처럼 인자까지 뽑아서 제안함.
    # "서울"을 영문으로 바꿔 넣는 것은 도구 설명(docstring)을 보고 판단한 결과.

    import requests
    import warnings
    warnings.filterwarnings("ignore")

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    # 1. 실제 날씨 API 도구 정의 (Open-Meteo 사용)

    @tool

    def get_weather(location: str) -> str:
        """특정 지역의 현재 날씨와 기온을 조회합니다.
        location: 조회할 도시 이름. (주의: API 검색을 위해 '서울'은 'Seoul'처럼 반드시 영문으로 번역해서 입력하세요)
        """
        try:
            # Step 1: 도시 이름을 위도 경도로 변환 (Geocoding API)

            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=ko"
            geo_resp = requests.get(geo_url).json()
            if "results" not in geo_resp:
                return f"{location}의 좌표를 찾을 수 없습니다. 도시 이름을 영어로 다시 시도해보세요."
            lat = geo_resp["results"][0]["latitude"]
            lon = geo_resp["results"][0]["longitude"]

            # Step 2: 위경도를 바탕으로 현재 기온 풍속 조회 (Forecast API)

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            weather_resp = requests.get(weather_url).json()
            current = weather_resp.get("current_weather", {})
            temp = current.get("temperature", "알 수 없음")
            windspeed = current.get("windspeed", "알 수 없음")
            return f"{location}의 현재 기온은 {temp}도이며, 풍속은 {windspeed}km/h입니다."
        
        except Exception as e:
            return f"날씨 정보를 불러오는 중 에러가 발생했습니다: {e}"

    # 2. 모델 초기화 및 도구 바인딩

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna", temperature=0)

    # bind_tools로 모델에게 "필요하면 이 도구를 써도 좋아"라고 JSON 스키마를 전달

    llm_with_tools = llm.bind_tools([get_weather])

    # 3. 모델 호출

    user_query = "지금 서울 날씨 어때? 덥진 않아?"
    response = llm_with_tools.invoke(user_query)

    # 4. Function Calling 결과 파싱 및 실행
    # 모델이 일반 텍스트 대신 함수 호출(tool_calls)을 제안했는지 확인

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        name = tool_call["name"]
        args = tool_call["args"]
        print(f"[LLM의 판단 (Function Calling)]")
        print(f"호출할 함수: {name}")
        print(f"추출된 인자: {args}\n")

        # 5. 애플리케이션 단에서 실제 파이썬 함수를 실행

        result = get_weather.invoke(args)
        print(f"[실측 데이터 결과]\n{result}")

    else:
        print(response.content)


    ## 09_02_03 펑션 콜링에 추론 교차를 얹기.

    # 인자가 2개인 도구는 pydantic 스키마로 설명(StructuredTool).
    # chat_history(이전 대화)와 agent_scratchpad(이번 질문의 도구 호출 과정)를 나눠서 멀티턴 대화를 기억함.
    

    from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_classic.tools import Tool, StructuredTool
    from pydantic import BaseModel, Field

    # 1. 도구(Tools) 정의 (실제 API 대신 고정된 문자열을 돌려주는 예시 도구)

    def get_stock_history(symbol: str, period: str):
        # 실제 환경에서는 DB나 API에서 기간별 데이터를 가져오는 로직이 들어갑니다.
        return f"{symbol}의 {period} 동안 주가는 96,500원에서 190,000원으로 96.9% 상승했습니다."

    def get_financial_news(symbol: str):
        return f"{symbol} 관련 뉴스: 차세대 AI 메모리 HBM4의 양산 소식과 가격 인상 협상에 따른 수익성 개선 기대감 고조."

    # 인자가 2개인 도구는 pydantic 스키마로 각 인자의 이름·설명을 알려줌.

    class StockHistoryInput(BaseModel):

        symbol: str = Field(description="주가 이력을 조회할 종목 코드 또는 이름")
        period: str = Field(description="주가 이력을 조회할 기간 (예: '최근 3개월', '1년')")

    tools = [
        StructuredTool(
            name="get_stock_history",
            func=get_stock_history,
            description="특정 종목(symbol)의 주어진 기간(period) 동안 주가 이력을 조회합니다.",
            args_schema=StockHistoryInput
        ),
        Tool(name="get_financial_news", func=get_financial_news,
             description="특정 종목의 최신 금융 뉴스를 가져옵니다.")
    ]

    # 2. 메모리가 포함된 프롬프트 설계

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 금융 데이터를 분석하는 전문 에이전트야. 도구를 사용하여 정확한 정보를 제공해."),
        MessagesPlaceholder(variable_name="chat_history"),      # 이전 턴들의 대화 (멀티턴)
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 이번 턴의 도구 호출 과정 (멀티스텝)
    ])

    # 3. 모델 및 에이전트 설정


    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna", temperature=0)

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4. 세션별 메모리 관리 (멀티턴의 핵심)

    demo_chat_history = ChatMessageHistory()

    conversational_agent_executor = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: demo_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # 5. 실행 (연쇄 사고 테스트)

    response = conversational_agent_executor.invoke(
        {"input": "최근 3개월 동안 삼성전자 주가 변동률과 관련 최신 뉴스 요약을 알려줘"},
        config={"configurable": {"session_id": "ch09_test_01"}}
    )

    print(response["output"])


    ## 09_02_04 여러 도구를 한 번에 병렬 제안

    # 질문 하나에 주가·환율·재무 도구 호출을 한 번에 여러 개 제안받아 차례로 실행    

    import yfinance as yf


    # 1. 실제 금융 데이터 API 도구 (Yahoo Finance 연동)

    @tool

    def get_stock_price(symbol: str) -> str:
        """특정 종목의 실시간 주가 정보를 조회합니다.
        주의: 한국 주식은 반드시 종목코드 뒤에 '.KS'를 붙여야 합니다 (예: 삼성전자 -> 005930.KS).
        """
        try:
            ticker = yf.Ticker(symbol)
            # fast_info를 사용하면 가볍고 빠르게 현재가를 가져옵니다.
            price = ticker.fast_info['last_price']
            return f"{symbol}의 현재 주가는 {price:,.0f}원입니다."
        except Exception as e:
            return f"주가 정보를 가져오는 데 실패했습니다. 심볼을 확인해주세요: {symbol}"

    @tool

    def get_exchange_rate(currency_pair: str = "USDKRW=X") -> str:
        """환율 정보를 조회합니다. 기본값은 USD/KRW 환율을 의미하는 'USDKRW=X'입니다."""
        try:
            ticker = yf.Ticker(currency_pair)
            rate = ticker.fast_info['last_price']
            return f"현재 {currency_pair} 환율은 {rate:,.2f}원입니다."
        except Exception as e:
            return f"환율 정보를 가져오는 데 실패했습니다: {currency_pair}"

    @tool

    def summarize_financials(symbol: str) -> str:
        """특정 기업의 재무 요약 정보(매출, 영업이익률 등)를 제공합니다.
        주의: 한국 주식은 종목코드 뒤에 '.KS'를 붙여야 합니다 (예: 삼성전자 -> 005930.KS).
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            revenue = info.get('totalRevenue', 0)
            margins = info.get('operatingMargins', 0) * 100
            # 매출을 보기 쉽게 조 단위로 변환
            if revenue > 0:
                revenue_str = f"{revenue / 1_000_000_000_000:.2f}조 원"
            else:
                revenue_str = "정보 없음"
            return f"{symbol}의 최근 매출은 {revenue_str}이며, 영업이익률은 약 {margins:.2f}%입니다."
        except Exception as e:
            return f"재무 정보를 불러오는 데 실패했습니다: {symbol}"


    # 2. 모델 초기화 및 도구 바인딩

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna", temperature=0)
    tools = [get_stock_price, get_exchange_rate, summarize_financials]
    llm_with_tools = llm.bind_tools(tools)


    # 3. 에이전트 실행 로직 (한 번의 응답에 tool_calls가 여러 개 들어올 수 있음)

    user_input = "삼성전자 주가와 현재 환율, 그리고 삼성전자의 재무 상태를 요약해줘."
    ai_msg = llm_with_tools.invoke(user_input)


    # 4. 도구 호출 제안 확인 및 실제 실행

    if ai_msg.tool_calls:

        print("=== [LLM의 도구 호출 제안 및 실행 결과] ===\n")

        for tool_call in ai_msg.tool_calls:

            # tool_call["name"]과 실제 함수 객체를 매핑

            selected_tool = {
                "get_stock_price": get_stock_price,
                "get_exchange_rate": get_exchange_rate,
                "summarize_financials": summarize_financials
            }[tool_call["name"].lower()]

            # 도구 실제 실행

            tool_output = selected_tool.invoke(tool_call["args"])
            print(f"[실행 도구]: {tool_call['name']}")
            print(f"[입력 파라미터]: {tool_call['args']}")
            print(f"[실측 결과]: {tool_output}\n")

    else:
        print(ai_msg.content)


def part_09_03():

    '''
    예외 처리와 병렬 실행: 도구 호출이 실패하거나 LLM이 인자를 잘못 줄 때 버티는 방법, 여러 도구를 동시에 실행하는 방법
    09_03_01 준비 / 09_03_02 유효성 검사(pydantic) / 09_03_03 재시도(tenacity) / 09_03_04 세 기법 통합 / 09_03_05 병렬 실행(asyncio)
    '''

    ## 09_03_01 준비
    # 09_03_05 병렬 실행에서 쓸 도구와 tool_calls(ai_msg) 만들기 (09_02_04와 같은 코드)
    # 09_02_04와 같은 도구 3개로 ai_msg(도구 호출 제안 3개)를 만들어 둠

    # import nest_asyncio; nest_asyncio.apply()   # .py에서는 필요 없음 (09_00 참고)

    import yfinance as yf

    import warnings
    warnings.filterwarnings("ignore")  # 불필요한 경고 메시지 숨김

    from langchain_openai import ChatOpenAI

    from langchain_core.tools import tool



    # 1. 실제 금융 데이터 API 도구 (Yahoo Finance 연동)

    @tool

    def get_stock_price(symbol: str) -> str:

        """
        특정 종목의 실시간 주가 정보를 조회합니다.
        주의: 한국 주식은 반드시 종목코드 뒤에 '.KS'를 붙여야 합니다 (예: 삼성전자 -> 005930.KS).
        """

        try:
            ticker = yf.Ticker(symbol)
            # fast_info를 사용하면 가볍고 빠르게 현재가를 가져옵니다.
            price = ticker.fast_info['last_price']
            return f"{symbol}의 현재 주가는 {price:,.0f}원입니다."
        except Exception as e:
            return f"주가 정보를 가져오는 데 실패했습니다. 심볼을 확인해주세요: {symbol}"


    @tool

    def get_exchange_rate(currency_pair: str = "USDKRW=X") -> str:
        """환율 정보를 조회합니다. 기본값은 USD/KRW 환율을 의미하는 'USDKRW=X'입니다."""
        try:
            ticker = yf.Ticker(currency_pair)
            rate = ticker.fast_info['last_price']
            return f"현재 {currency_pair} 환율은 {rate:,.2f}원입니다."
        except Exception as e:
            return f"환율 정보를 가져오는 데 실패했습니다: {currency_pair}"

    @tool

    def summarize_financials(symbol: str) -> str:
        """특정 기업의 재무 요약 정보(매출, 영업이익률 등)를 제공합니다.
        주의: 한국 주식은 종목코드 뒤에 '.KS'를 붙여야 합니다 (예: 삼성전자 -> 005930.KS).
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            revenue = info.get('totalRevenue', 0)
            margins = info.get('operatingMargins', 0) * 100
            # 매출을 보기 쉽게 조 단위로 변환
            if revenue > 0:
                revenue_str = f"{revenue / 1_000_000_000_000:.2f}조 원"
            else:
                revenue_str = "정보 없음"
            return f"{symbol}의 최근 매출은 {revenue_str}이며, 영업이익률은 약 {margins:.2f}%입니다."
        except Exception as e:
            return f"재무 정보를 불러오는 데 실패했습니다: {symbol}"


    # 2. 모델 초기화 및 도구 바인딩

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-6-luna", temperature=0)

    tools = [get_stock_price, get_exchange_rate, summarize_financials]

    llm_with_tools = llm.bind_tools(tools)


    # 3. 에이전트 실행 로직 (여기서 받은 ai_msg.tool_calls를 09_03_05에서 동시에 실행)

    user_input = "삼성전자 주가와 현재 환율, 그리고 삼성전자의 재무 상태를 요약해줘."

    ai_msg = llm_with_tools.invoke(user_input)


    # 09_03_02 유효성 검사(pydantic) 레이어 구축

    # LLM이 넘긴 인자를 실행 전에 검사. 소문자 종목 코드면 ValueError.

    from pydantic import BaseModel, Field, field_validator

    class StockQuerySchema(BaseModel):

        symbol: str = Field(..., description="종목 코드")

        @field_validator("symbol")

        @classmethod

        def validate_symbol(cls, v: str):
            if not v.isupper():
                raise ValueError("종목 코드는 반드시 대문자여야 합니다.")
            return v


    # 09_03_03 Tenacity를 이용한 재시도 메커니즘
    # 일시적인 네트워크 오류는 4~10초 기다렸다가 최대 3번 다시 시도
    # 주소는 예시

    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(
            
        stop=stop_after_attempt(3),                      # 최대 3번까지 재시도
        wait=wait_exponential(multiplier=1, min=4, max=10),  # 대기 시간을 지수적으로 증가
        reraise=True                                     # 3번 모두 실패하면 원래 예외 발생
    )

    def call_external_api(query):
        response = requests.get(f"https://api.finance.com/v1/{query}")
        response.raise_for_status()
        return response.json()

    # 자율적 오류 수정(Self-Correction) 루프 (추가)

    # 도구가 예외를 던지는 대신 "무엇이 틀렸는지" 문장을 돌려주면, 그 문장이 에이전트의 Observation이 되어
    # LLM이 인자를 고쳐서 다시 호출한다 (09_03_04의 except 블록이 이 역할).


    ## 09_03_04 세 가지 기법의 통합 (검증 + 재시도 + 오류 메시지 반환)

    # 검증 → 재시도 → 실패하면 예외를 던지지 않고 "무엇이 틀렸는지" 문장을 돌려줌.
    # 에이전트 안에서는 이 문장이 Observation이 되어 LLM이 인자를 고쳐 다시 호출함(Self-Correction).

    from langchain_classic.tools import StructuredTool

    class StockQuerySchema(BaseModel):
        symbol: str = Field(..., description="종목 코드 (반드시 대문자, 예: AAPL, 005930.KS)")

        @field_validator("symbol")

        @classmethod

        def validate_symbol(cls, v: str):
            if not v.isupper():
                raise ValueError(f"종목 코드 '{v}'는 반드시 대문자여야 합니다. 예: AAPL, 005930.KS")
            return v

    @retry(
        stop=stop_after_attempt(3),                      # 최대 3번 재시도
        wait=wait_exponential(multiplier=1, min=4, max=10),  # 대기 간격을 지수적으로 늘리되 4~10초로 제한
        reraise=True                                     # 3번 모두 실패 시 원래 예외 발생
    )

    def _fetch_stock_price_from_api(symbol: str) -> dict:
        """실제 외부 API를 호출하는 내부 함수. 에러 시 자동 재시도합니다."""
        response = requests.get(
            f"https://api.finance.com/v1/quote?symbol={symbol}", timeout=5
        )
        response.raise_for_status()
        return response.json()

    def get_stock_price_safe(symbol: str) -> str:
        """Pydantic 검증 + Tenacity 재시도가 적용된 주가 조회 함수"""
        try:
            # 1단계: Pydantic으로 입력값 검증
            validated = StockQuerySchema(symbol=symbol)
            # 2단계: Tenacity 재시도 보호 하에 API 호출
            data = _fetch_stock_price_from_api(validated.symbol)
            price = data.get("price", "알 수 없음")
            return f"{validated.symbol}의 현재 주가는 {price}원입니다."
        
        except ValueError as e:
            # Pydantic 검증 실패: LLM이 잘못된 형식으로 인자를 넘긴 경우
            # 이 메시지가 Observation으로 에이전트에 전달되어 Self-Correction을 트리거합니다.
            return f"입력값 오류: {e} 종목 코드를 대문자로 다시 입력해주세요."
        
        except Exception as e:
            # API 호출 최종 실패 (3번 재시도 후에도 실패)
            return f"API 호출 실패: {e}. 잠시 후 다시 시도하거나 다른 종목을 조회해주세요."

    # AgentExecutor에 등록할 때는 StructuredTool로 래핑

    safe_stock_tool = StructuredTool(
        name="get_stock_price_safe",
        func=get_stock_price_safe,
        description="특정 종목의 현재 주가를 조회합니다. 종목 코드는 반드시 대문자로 입력하세요 (예: AAPL, 005930.KS).",
        args_schema=StockQuerySchema
    )

    # (추가) 동작 확인: 책은 정의만 하고 호출하지 않으므로 직접 불러 본다 (OpenAI API 호출 없음)

    # ① 소문자 → pydantic 검증에서 바로 걸려 "입력값 오류" 문장을 돌려줌 (네트워크 호출 없음)
    print(get_stock_price_safe("aapl"))

    # ② 대문자 → 검증 통과 후 예시 주소(api.finance.com) 호출이 실패 → 4초씩 기다리며 3번 시도한 뒤 "API 호출 실패"
    print(get_stock_price_safe("AAPL"))


    ## 09_03_05 병렬 실행(asyncio)

    # 09_02_04는 tool_calls를 for 문으로 하나씩 실행. 여기서는 동시에 실행.
    # 09_02_04는 도구를 하나씩 실행했지만, 여기서는 asyncio.gather로 세 도구를 동시에 실행.
    # 동기 함수인 yfinance는 asyncio.to_thread로 별도 스레드에서 실행

    import asyncio

    selected_tool = {
        "get_stock_price": get_stock_price,
        "get_exchange_rate": get_exchange_rate,
        "summarize_financials": summarize_financials,
    }

    async def call_one(tool_call):

        tool = selected_tool[tool_call["name"].lower()]
        # yfinance 호출은 동기(blocking) 함수이므로, 별도 스레드에서 실행해
        # 이벤트 루프를 막지 않도록 asyncio.to_thread로 감쌉니다.

        return await asyncio.to_thread(tool.invoke, tool_call["args"])

    async def run_parallel(ai_msg):

        # gather가 모든 호출을 동시에 시작하고, 전부 끝날 때까지 기다립니다.

        return await asyncio.gather(*[call_one(tc) for tc in ai_msg.tool_calls])

    # results = await run_parallel(ai_msg)  # 노트북에서는 nest_asyncio 덕분에 await를 바로 사용

    results = asyncio.run(run_parallel(ai_msg))  
    # .py에는 최상위 await가 없으므로 asyncio.run으로 실행

    for r in results:
        print(r)


def part_09_04():
    ## 09_04_01 Text-to-SQL (작성 예정, hospital.db는 실행할 때 새로 만들어지므로 .gitignore)

    print("09_04: 아직 작성 전")


PARTS = {"09_01": part_09_01, "09_02": part_09_02, "09_03": part_09_03, "09_04": part_09_04}


## 09_99 실행 로그: 파트마다 화면 출력 전체를 logs/09/<파트>_<시각>.log에도 저장 (분석용, .gitignore)

import re, datetime, contextlib, traceback

class _Tee:
    """화면과 로그 파일에 동시에 쓴다. 파일에는 verbose 출력의 색상 코드(ANSI)를 지워서 저장한다."""
    _ansi = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(self._ansi.sub("", s))
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


if __name__ == "__main__":
    # for name in (sys.argv[1:] or PARTS):
    #     print(f"\n{'=' * 20} {name} {'=' * 20}")
    #     PARTS[name]()
    os.makedirs("logs/09", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in (sys.argv[1:] or PARTS):
        path = f"logs/09/{name}_{stamp}.log"
        with open(path, "w", encoding="utf-8") as log, \
             contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
             contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            print(f"\n{'=' * 20} {name} ({stamp}) {'=' * 20}")
            try:
                PARTS[name]()
            except Exception:
                traceback.print_exc()   # 오류도 로그에 남기고 다음 파트로 넘어감
        print(f"[로그 저장] {path}")
