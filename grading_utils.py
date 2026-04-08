import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


evaluation_items = [
    {
        "question": "금융투자회사 직원에게 계좌관리를 일임할 때 투자일임계약서에 세부적으로 명기해야 한다고 한 두 가지는 무엇인가요?",
        "answer": "직원에게 일임을 하는 사항, 직원이 해서는 안되는 사항",
        "source_page": 38,
    },
    {
        "question": "당사자 간의 분쟁해결이 가장 손쉽고 합리적일 수 있다고 설명하면서도, 자본시장법 제55조와 관련해 유의하라고 한 금지 행위는 무엇인가요?",
        "answer": "손실보전행위",
        "source_page": 41,
    },
    {
        "question": "발간사에서 최근 주요 이슈를 반영해 기존의 무엇을 보강한 증보판을 발간했다고 소개하나요?",
        "answer": "분쟁조정사례·판례 핸드북",
        "source_page": 4,
    },
    {
        "question": "발간사에서 새롭게 추가했다고 소개한 중요 정보 두 가지는 무엇인가요?",
        "answer": "금융사기에 대한 예방·대처법, 모바일 폰을 활용한 투자 시 유의사항",
        "source_page": 4,
    },
    {
        "question": "공시정보 체크요령에서 증권회사 안내가 없어도 스스로 확인해야 한다고 든 CB 투자 관련 권리는 무엇인가요?",
        "answer": "전환권 행사시점과 매수청구권",
        "source_page": 27,
    },
    {
        "question": "투자자유형 분류기준의 자세한 내용은 한국금융투자협회 홈페이지의 어떤 경로에서 확인할 수 있나요?",
        "answer": "www.kofia.or.kr → 법규정보 시스템 → 모범규준",
        "source_page": 30,
    },
    {
        "question": "펀드 가입 절차 설명에서 '표준투자권유준칙'에 따라 투자자를 5등급 이상으로 분류할 때 제시한 다섯 유형은 무엇인가요?",
        "answer": "안정형, 안정추구형, 위험·수익중립형, 적극투자형, 공격투자형",
        "source_page": 30,
    },
    {
        "question": "'투자자별 펀드잔고 통보' 서비스는 어떤 두 가지 정보를 어떤 방식으로 제공하며, 언제 제공받지 않을 수 있나요?",
        "answer": "펀드잔고와 개별 수익률 정보를 전자우편 등으로 제공하며, 제공을 원하지 않을 경우 별도 신청을 통해 제공받지 않을 수 있습니다",
        "source_page": 31,
    },
    {
        "question": "자산보관·관리보고서에는 어떤 네 가지 정보가 제공된다고 설명하나요?",
        "answer": "펀드매니저 변동사항, 약관 변경 사항, 수익자 총회 의결 내용, 운용상 위반사항·시정사항",
        "source_page": 32,
    },
    {
        "question": "'모바일폰 금융거래 10계명' 중 6번과 9번의 핵심 내용은 무엇인가요?",
        "answer": "휴대폰 문자서비스(SMS), 일회용비밀번호(OTP) 이용하기 / 잠금기능을 설정하고 잠금비밀번호는 수시로 변경하기",
        "source_page": 26,
    },
    {
        "question": "임의매매에 따른 처벌 조항 설명에서 제시한 형사처벌 수준과 조문 번호는 무엇인가요?",
        "answer": "5년 이하의 징역 또는 2억원 이하의 벌금, §444",
        "source_page": 52,
    },
    {
        "question": "임의매매 설명에서 손해배상금액 결정 시 과실상계 예시로 든 고객 과실 두 가지는 무엇인가요?",
        "answer": "장기간 거래내역 확인 소홀, 계좌번호·비밀번호 관리 소홀",
        "source_page": 52,
    },
    {
        "question": "선물환 계약 불완전판매 사례 15에서 법원이 금융회사의 손해배상 비율을 전체 손해액의 몇 퍼센트로 본다고 판단했나요?",
        "answer": "40%",
        "source_page": 102,
    },
    {
        "question": "금융투자상품 불완전판매 사례 17에서 투자자 A와 D의 투자금액은 각각 얼마인가요?",
        "answer": "A는 2억 4,300만원, D는 1억 5,100여만원",
        "source_page": 104,
    },
    {
        "question": "반대매매 대상종목 선정 기준으로 제시한 세 가지 예시는 무엇인가요?",
        "answer": "최근 매수 종목, 종목코드 번호 선(후)순위, 종목명 가나다순",
        "source_page": 115,
    },
    {
        "question": "반대매매 예시에서 195주가 7,000원에 체결된 것으로 가정할 경우 임의처분 금액은 약 얼마이며, 담보부족금액의 몇 배 수준인가요?",
        "answer": "약 137만원, 4.6배",
        "source_page": 115,
    },
    {
        "question": "피싱사기로 주민등록번호 등 개인정보를 알려줬을 경우 가까운 은행에 등록 요청하라고 한 시스템 이름은 무엇인가요?",
        "answer": "개인정보 노출자 사고예방 시스템",
        "source_page": 141,
    },
    {
        "question": "보이스피싱 피해금 환급절차에서 금감원의 공고 기간과 피해금 환급 소요 기간은 각각 얼마인가요?",
        "answer": "2개월간 공고, 3개월 정도 소요",
        "source_page": 141,
    },
    {
        "question": "대포폰 개설여부 조회는 어느 사이트에서 할 수 있으며, 함께 제공되는 서비스는 무엇인가요?",
        "answer": "www.msafer.or.kr, 명의도용 사전차단 서비스",
        "source_page": 143,
    },
    {
        "question": "유사투자자문업 관련 판례 사례에서 B 증권투자 전문가가 배분 받은 비율과 유료회원의 월 가입료는 각각 얼마인가요?",
        "answer": "가입료의 50%, 월 80만원 상당",
        "source_page": 145,
    },
]

answer_key = [item["answer"] for item in evaluation_items]


def grade_predictions(questions, predicted_answers, model_name="gpt-5-nano"):
    def normalize_for_match(text: str) -> str:
        text = (text or "").replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*/\s*", " / ", text)
        text = re.sub(r"\s*→\s*", " → ", text)
        text = re.sub(r"§\s*", "§", text)
        text = re.sub(r"\s*%\s*", "%", text)
        return text.strip()

    def is_exact_no_info(text: str) -> bool:
        stripped = normalize_for_match(text).strip(" .,:;!?\"'`()[]{}")
        return stripped == "없는 정보"

    def has_format_violation(predicted: str, answer: str):
        raw = (predicted or "").strip()
        pred = normalize_for_match(predicted)
        ans = normalize_for_match(answer)

        if not raw:
            return True, "빈 답변"
        if pred == ans:
            return False, ""
        if "\n" in raw or "\r" in raw:
            return True, "줄바꿈이나 여러 줄 형식"
        if re.search(r"(?m)^\s*[-*•]", raw) or re.search(r"(?m)^\s*\d+[.)]", raw):
            return True, "불릿이나 번호 형식"
        if re.match(r"^(정답|답변|답)\s*[:：]", raw):
            return True, "접두어"
        if ans and ans in pred and pred != ans:
            return True, "정답 외 설명이나 접미어"
        return False, ""

    grading_prompt = ChatPromptTemplate.from_template(
        """
너는 RAG 대회 자동 채점 시스템이다.
아래에는 질문, 모델 예측 답변, 정답이 주어진다.

채점 기준:
- 정답 문자열을 형식까지 정확히 맞추면 1점
- 의미가 맞더라도 정답 뒤에 설명, 조사, 접미어(예: "입니다"), 불릿, 번호, 줄바꿈이 붙으면 0점
- 숫자, 날짜, 코드, URL, "없는 정보" 같은 답은 정확해야 한다
- 예측 답변이 정확히 "없는 정보"인데 정답이 존재하면 반드시 0점이다
- 답에 여러 요소가 필요한 문항은 주요 요소를 모두 맞혀야 1점이다
- 형식은 맞았지만 일부 요소가 빠졌거나 표현이 덜 정확하면 0.5점
- 틀리거나 문서에 없는 내용을 지어내면 0점

반드시 아래 형식만 사용한다.
1번: 1점 (설명)
2번: 0.5점 (설명)
3번: 0점 (설명)

질문-답변 리스트:
{qa_pairs}
"""
    )

    forced_results = {}
    qa_rows = []

    for idx in range(len(questions)):
        predicted = predicted_answers[idx]
        answer = answer_key[idx]
        number = idx + 1
        normalized_predicted = normalize_for_match(predicted)
        normalized_answer = normalize_for_match(answer)

        if normalized_predicted == normalized_answer:
            forced_results[number] = (
                f"{number}번: 1점 (설명: 출력 형식과 내용이 정답과 정확히 일치합니다)"
            )
            continue

        if is_exact_no_info(predicted) and not is_exact_no_info(answer):
            forced_results[number] = (
                f"{number}번: 0점 (설명: 예측이 '없는 정보'인데 정답은 문서에 존재함)"
            )
            continue

        format_violation, reason = has_format_violation(predicted, answer)
        if format_violation:
            forced_results[number] = (
                f"{number}번: 0점 (설명: 형식 위반 사유: {reason})"
            )
            continue

        qa_rows.append(
            f"{number}. 질문: {questions[idx]}\n예측: {predicted}\n정답: {answer}"
        )

    llm_result = ""
    if qa_rows:
        llm = ChatOpenAI(model=model_name, temperature=0)
        parser = StrOutputParser()
        grading_chain = grading_prompt | llm | parser
        llm_result = grading_chain.invoke({"qa_pairs": "\n".join(qa_rows)})

    for line in llm_result.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        prefix = stripped.split("번:", 1)[0]
        if prefix.isdigit():
            forced_results[int(prefix)] = stripped

    return "\n".join(forced_results[idx] for idx in sorted(forced_results))
