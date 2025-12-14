from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional

from .gemini_client import GeminiClient


class QuestionCategory(str, Enum):
    """질문 카테고리"""
    MOTIVATION = "지원동기"
    PROJECT_SUCCESS = "프로젝트 경험"
    TECHNICAL = "기술/역량"
    COLLABORATION = "협업/커뮤니케이션"
    PROBLEM_SOLVING = "문제해결"
    FAILURE_OVERCOME = "실패극복"
    GROWTH = "성장가능성"


class Grade(str, Enum):
    """평가 등급"""
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class Question:
    """면접 질문"""
    id: int
    category: str
    content: str
    time_limit_seconds: int = 120  # 기본 2분


@dataclass
class Answer:
    """사용자 답변"""
    question_id: int
    content: str
    duration_seconds: Optional[int] = None
    followup_question: Optional[str] = None
    followup_answer: Optional[str] = None


@dataclass
class STARScore:
    """STAR 기법 점수 (각 1-10점)"""
    situation: int
    task: int
    action: int
    result: int

    @property
    def total(self) -> int:
        return self.situation + self.task + self.action + self.result


@dataclass
class AdditionalScore:
    """추가 평가 점수 (각 1-10점)"""
    logic: int           # 논리성
    specificity: int     # 구체성
    job_relevance: int   # 직무관련성
    time_balance: int    # 시간분배

    @property
    def total(self) -> int:
        return self.logic + self.specificity + self.job_relevance + self.time_balance


@dataclass
class QuestionFeedback:
    """질문별 상세 피드백"""
    question_id: int
    question_content: str
    user_answer: str
    star_score: STARScore
    additional_score: AdditionalScore
    total_score: float
    strengths: List[str]
    improvements: List[str]
    example_answer: str


class InterviewAIService:
    """모의면접 AI 서비스 (Gemini 사용)"""

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()
        self._sessions: dict[str, dict] = {}

    # ----------------------------------------
    # 공통: Gemini 호출
    # ----------------------------------------
    def _generate(self, prompt: str) -> str:
        """Gemini API 호출 공통 래퍼"""
        return self.client.generate(prompt)

    # ----------------------------------------
    # 1) 질문 생성
    # ----------------------------------------
    def generate_questions(
        self,
        resume_text: str,
        job_position: str,
    ) -> tuple[str, List[Question]]:
        """자기소개서 기반 7개 맞춤형 면접 질문 생성"""
        prompt = self._build_question_prompt(resume_text, job_position)
        response_text = self._generate(prompt)
        questions = self._parse_questions(response_text)

        interview_id = str(uuid.uuid4())
        self._sessions[interview_id] = {
            "job_position": job_position,
            "resume_text": resume_text,
            "questions": questions,
        }
        return interview_id, questions

    def _build_question_prompt(self, resume_text: str, job_position: str) -> str:
        return f"""당신은 {job_position} 포지션 면접관입니다.
아래 자기소개서를 읽고, 실제 면접에서 물어볼 법한 심층 질문 7개를 생성하세요.

## 질문 생성 규칙
1. 모든 질문은 서로 다른 주제/카테고리여야 합니다.
2. 자기소개서 내용을 구체적으로 언급하며 질문하세요.
3. 구체적인 경험, 상황, 행동, 결과를 자연스럽게 물어보는 심층 질문으로 작성하세요.
4. 단답형이 아닌 구술형 답변을 유도하는 질문으로 작성하세요.
5. 질문에 'STAR 기법', 'STAR 방식' 등 특정 답변 형식을 요구하는 표현을 절대 사용하지 마세요.

## 카테고리 가이드 (참고용, 반드시 7개 모두 사용할 필요 없음)
자기소개서 내용에 맞게 아래 카테고리 중 적절한 것을 선택하세요:
- 지원동기: 해당 직무/회사에 대한 관심과 이해도
- 프로젝트 경험: 성공적으로 완수한 프로젝트 경험
- 기술/역량: 직무 관련 기술적 역량 심화
- 협업/커뮤니케이션: 팀워크, 갈등 해결 경험
- 문제해결: 어려운 상황을 해결한 경험
- 실패극복: 실패/약점을 극복한 사례
- 성장가능성: 미래 계획, 자기개발 의지

## 답변 시간 설정 기준
각 질문에 대해 적절한 답변 시간(초)을 설정하세요:
- 단순 사실/의견 질문 (예: 지원동기, 포부): 60-90초
- 경험 기반 질문 (구체적 상황 설명 필요): 120-150초
- 복합적 질문 (여러 단계 설명, 문제해결 과정): 150-180초

## 자기소개서
{resume_text}

## 지원 직무
{job_position}

## 출력 형식 (반드시 아래 JSON 형식으로만 출력하세요)
```json
[
  {{"id": 1, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 90}},
  {{"id": 2, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 150}},
  {{"id": 3, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 120}},
  {{"id": 4, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 150}},
  {{"id": 5, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 180}},
  {{"id": 6, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 120}},
  {{"id": 7, "category": "카테고리명", "content": "질문 내용...", "time_limit_seconds": 90}}
]
```"""

    def _parse_questions(self, response_text: str) -> List[Question]:
        """LLM 응답 텍스트에서 질문 리스트 JSON 파싱"""
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(
                    f"질문 생성 결과를 파싱할 수 없습니다: {response_text[:200]}"
                )

        questions_data = json.loads(json_str)
        return [
            Question(
                id=q["id"],
                category=q["category"],
                content=q["content"],
                time_limit_seconds=q.get("time_limit_seconds", 120),  # 기본값 120초
            )
            for q in questions_data
        ]

    # ----------------------------------------
    # 2) 꼬리질문 생성 (Follow-up Question)
    # ----------------------------------------
    @dataclass
    class FollowUpResult:
        """꼬리질문 결과"""
        has_followup: bool
        followup_question: Optional[str] = None
        time_limit_seconds: int = 60
        reason: Optional[str] = None

    def generate_followup_question(
        self,
        interview_id: str,
        question: Question,
        answer: Answer,
    ) -> Dict:
        """사용자 답변을 분석하여 필요 시 꼬리질문 생성
        
        Args:
            interview_id: 면접 세션 ID
            question: 원래 질문
            answer: 사용자 답변
            
        Returns:
            {
                "has_followup": bool,
                "followup_question": str or None,
                "time_limit_seconds": int,
                "reason": str or None
            }
        """
        session = self._sessions.get(interview_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {interview_id}")
        
        job_position = session["job_position"]
        prompt = self._build_followup_prompt(question, answer, job_position)
        response_text = self._generate(prompt)
        
        return self._parse_followup_response(response_text)
    
    def _build_followup_prompt(
        self, 
        question: Question, 
        answer: Answer, 
        job_position: str
    ) -> str:
        return f"""당신은 {job_position} 포지션 면접관입니다.
지원자의 답변을 분석하고, 필요한 경우 꼬리질문을 생성하세요.

## 꼬리질문이 필요한 경우
1. 답변이 너무 짧거나 추상적인 경우 (구체적인 사례 요청)
2. 주장만 있고 근거/예시가 부족한 경우
3. 경험을 언급했지만 본인의 역할이 불분명한 경우
4. 결과/성과에 대한 구체적 수치가 없는 경우
5. 흥미로운 내용이 있어 더 깊이 파고들고 싶은 경우

## 꼬리질문이 필요하지 않은 경우  
1. 답변이 충분히 구체적이고 완성도가 높은 경우
2. STAR 구조가 잘 갖춰진 경우

## 원래 질문
카테고리: {question.category}
내용: {question.content}

## 지원자 답변
{answer.content}

## 출력 규칙
- 꼬리질문이 필요하면 "has_followup": true
- 불필요하면 "has_followup": false
- 꼬리질문은 압박형이 아닌 탐색형으로 ("그 부분을 좀 더 자세히 말씀해주시겠어요?")
- 질문에 'STAR 기법' 등 특정 형식 요구 표현 사용 금지

## 출력 형식 (반드시 아래 JSON 형식으로만 출력하세요)
```json
{{
  "has_followup": true,
  "followup_question": "꼬리질문 내용...",
  "time_limit_seconds": 60,
  "reason": "꼬리질문이 필요한 이유 (내부용, 사용자에게 보여주지 않음)"
}}
```

또는 꼬리질문이 불필요할 경우:
```json
{{
  "has_followup": false,
  "followup_question": null,
  "time_limit_seconds": 0,
  "reason": "답변이 충분히 구체적입니다."
}}
```"""

    def _parse_followup_response(self, response_text: str) -> Dict:
        """꼬리질문 응답 파싱"""
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 파싱 실패시 꼬리질문 없음으로 처리
                return {
                    "has_followup": False,
                    "followup_question": None,
                    "time_limit_seconds": 0,
                    "reason": "응답 파싱 실패"
                }
        
        try:
            data = json.loads(json_str)
            return {
                "has_followup": data.get("has_followup", False),
                "followup_question": data.get("followup_question"),
                "time_limit_seconds": data.get("time_limit_seconds", 60),
                "reason": data.get("reason")
            }
        except json.JSONDecodeError:
            return {
                "has_followup": False,
                "followup_question": None,
                "time_limit_seconds": 0,
                "reason": "JSON 파싱 실패"
            }

    # ----------------------------------------
    # 3) 7개 답변 한 번에 평가
    # ----------------------------------------
    def evaluate_answers(self, interview_id: str, answers: List[Answer]) -> Dict:
        """7개 답변에 대한 종합 평가 수행"""
        session = self._sessions.get(interview_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {interview_id}")

        job_position = session["job_position"]
        questions: List[Question] = session["questions"]

        feedbacks: List[QuestionFeedback] = []
        for ans in answers:
            question = next(q for q in questions if q.id == ans.question_id)
            fb = self._evaluate_single_answer(question, ans, job_position)
            feedbacks.append(fb)

        return self._generate_comprehensive_feedback(interview_id, job_position, feedbacks)

    def _evaluate_single_answer(
        self,
        question: Question,
        answer: Answer,
        job_position: str,
    ) -> QuestionFeedback:
        """단일 답변 평가"""
        
        # 꼬리질문/답변 정보 구성
        followup_info = ""
        if answer.followup_question and answer.followup_answer:
            followup_info = f"""
### 꼬리질문 및 답변
꼬리질문: {answer.followup_question}
꼬리답변: {answer.followup_answer}
"""

        prompt = f"""당신은 {job_position} 채용 면접 평가 전문가입니다.
아래 면접 질문과 지원자의 답변을 STAR 기법 기준으로 상세히 평가하세요.
만약 꼬리질문과 답변이 있다면, 이를 포함하여 종합적으로 평가하세요.

## 평가 기준 (각 항목 1-10점)

### STAR 기법 평가
- Situation (상황): 상황과 배경 설명의 명확성
- Task (과제): 본인의 역할과 목표 정의의 명확성
- Action (행동): 구체적인 행동과 노력의 상세함
- Result (결과): 성과와 학습 포인트의 구체성

### 추가 평가
- 논리성: 답변 구조의 논리적 흐름과 일관성
- 구체성: 수치, 사례, 예시의 구체적 제시 (꼬리질문을 통해 보강되었는지 확인)
- 직무관련성: {job_position} 직무와의 연관성
- 시간분배: 답변 길이의 적절성 (이상적: 1-2분, 150-300자)

## 질문
카테고리: {question.category}
내용: {question.content}

## 지원자 답변
{answer.content}
{followup_info}

## 출력 형식 (반드시 아래 JSON 형식으로만 출력하세요)
```json
{{
  "star_score": {{
    "situation": 7,
    "task": 6,
    "action": 8,
    "result": 5
  }},
  "additional_score": {{
    "logic": 7,
    "specificity": 6,
    "job_relevance": 8,
    "time_balance": 7
  }},
  "strengths": [
    "잘한 점 1",
    "잘한 점 2"
  ],
  "improvements": [
    "개선 포인트 1",
    "개선 포인트 2"
  ],
  "example_answer": "이 질문에 대한 모범 답변 예시 (200자 내외)"
}}
```"""
        response_text = self._generate(prompt)

        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        data = json.loads(json_str)

        star_score = STARScore(**data["star_score"])
        additional_score = AdditionalScore(**data["additional_score"])

        raw_total = star_score.total + additional_score.total
        total_score = round((raw_total / 80) * 100, 1)

        return QuestionFeedback(
            question_id=question.id,
            question_content=question.content,
            user_answer=answer.content,
            star_score=star_score,
            additional_score=additional_score,
            total_score=total_score,
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            example_answer=data.get("example_answer", ""),
        )

    # ----------------------------------------
    # 3) 종합 피드백
    # ----------------------------------------
    def _calculate_grade(self, score: float) -> Grade:
        if score >= 90:
            return Grade.S
        elif score >= 80:
            return Grade.A
        elif score >= 70:
            return Grade.B
        elif score >= 60:
            return Grade.C
        else:
            return Grade.D

    def _generate_comprehensive_feedback(
        self,
        interview_id: str,
        job_position: str,
        feedbacks: List[QuestionFeedback],
    ) -> Dict:
        if not feedbacks:
            raise ValueError("feedbacks 가 비어 있습니다.")

        n = len(feedbacks)

        overall_score = round(sum(f.total_score for f in feedbacks) / n, 1)
        overall_grade = self._calculate_grade(overall_score)

        star_averages = {
            "situation": round(sum(f.star_score.situation for f in feedbacks) / n, 1),
            "task": round(sum(f.star_score.task for f in feedbacks) / n, 1),
            "action": round(sum(f.star_score.action for f in feedbacks) / n, 1),
            "result": round(sum(f.star_score.result for f in feedbacks) / n, 1),
        }

        additional_averages = {
            "logic": round(sum(f.additional_score.logic for f in feedbacks) / n, 1),
            "specificity": round(sum(f.additional_score.specificity for f in feedbacks) / n, 1),
            "job_relevance": round(sum(f.additional_score.job_relevance for f in feedbacks) / n, 1),
            "time_balance": round(sum(f.additional_score.time_balance for f in feedbacks) / n, 1),
        }

        summary_prompt = self._build_summary_prompt(
            job_position,
            feedbacks,
            star_averages,
            additional_averages,
            overall_score,
        )
        summary_response = self._generate(summary_prompt)

        json_match = re.search(r"```json\s*(.*?)\s*```", summary_response, re.DOTALL)
        if json_match:
            summary_data = json.loads(json_match.group(1))
        else:
            summary_data = {
                "overall_strengths": ["전반적으로 성실한 답변입니다."],
                "overall_improvements": ["구체적인 수치와 사례를 더 제시해 보세요."],
                "final_advice": f"{job_position} 직무와 직접적으로 연결되는 경험 위주로 답변을 구성해 보세요.",
            }

        return {
            "interview_id": interview_id,
            "job_position": job_position,
            "overall_score": overall_score,
            "overall_grade": overall_grade.value,
            "star_averages": star_averages,
            "additional_averages": additional_averages,
            "question_feedbacks": [self._feedback_to_dict(fb) for fb in feedbacks],
            "overall_strengths": summary_data.get("overall_strengths", []),
            "overall_improvements": summary_data.get("overall_improvements", []),
            "final_advice": summary_data.get("final_advice", ""),
        }

    def _build_summary_prompt(
        self,
        job_position: str,
        feedbacks: List[QuestionFeedback],
        star_avg: Dict[str, float],
        add_avg: Dict[str, float],
        score: float,
    ) -> str:
        lines = []
        for f in feedbacks:
            s = ", ".join(f.strengths[:2]) if f.strengths else "강점 미기재"
            im = ", ".join(f.improvements[:2]) if f.improvements else "개선점 미기재"
            lines.append(
                f"Q{f.question_id}. 점수: {f.total_score}점, 강점: {s}, 개선점: {im}"
            )
        feedback_summary = "\n".join(lines)

        return f"""당신은 {job_position} 채용 면접 평가 전문가입니다.
지원자의 7개 면접 답변 평가 결과를 종합하여 최종 피드백을 작성하세요.

## 평가 결과 요약
- 종합 점수: {score}점

### STAR 항목별 평균
- Situation: {star_avg['situation']}점 / Task: {star_avg['task']}점 / Action: {star_avg['action']}점 / Result: {star_avg['result']}점

### 추가 평가 항목별 평균
- 논리성: {add_avg['logic']}점 / 구체성: {add_avg['specificity']}점 / 직무관련성: {add_avg['job_relevance']}점 / 시간분배: {add_avg['time_balance']}점

### 질문별 요약
{feedback_summary}

## 출력 형식 (반드시 아래 JSON 형식으로만 출력하세요)
```json
{{
  "overall_strengths": ["전반적인 강점 1", "전반적인 강점 2", "전반적인 강점 3"],
  "overall_improvements": ["전반적인 개선점 1 (구체적 방법 포함)", "전반적인 개선점 2", "전반적인 개선점 3"],
  "final_advice": "{job_position} 직무 준비를 위한 200자 내외 종합 조언"
}}
```"""

    @staticmethod
    def _feedback_to_dict(f: QuestionFeedback) -> Dict:
        return {
            "question_id": f.question_id,
            "question_content": f.question_content,
            "user_answer": f.user_answer,
            "star_score": asdict(f.star_score),
            "additional_score": asdict(f.additional_score),
            "total_score": f.total_score,
            "strengths": f.strengths,
            "improvements": f.improvements,
            "example_answer": f.example_answer,
        }
