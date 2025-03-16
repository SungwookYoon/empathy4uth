import os
import json
from typing import List, Dict, Any
from src.config.api_config import client, GPT_MODEL, MAX_TOKENS, TEMPERATURE, validate_api_key
import sys

class GPTTester:
    def __init__(self):
        try:
            validate_api_key()
            self.client = client
        except Exception as e:
            print(f"초기화 오류: {str(e)}")
            sys.exit(1)
        self.model = GPT_MODEL

    def test_counseling_analysis(self, text: str) -> Dict[str, Any]:
        """상담 데이터 분석 테스트"""
        try:
            print("\n=== 상담 데이터 분석 테스트 ===")
            prompt = f"""다음 청소년 상담 텍스트를 분석하여 위기 단계를 판단하고 적절한 개입 방안을 제시해주세요:

텍스트: {text}

다음 형식으로 응답해주세요:
- 위기 단계: [정상/관찰필요/상담필요/위험/긴급]
- 주요 감정:
- 핵심 문제:
- 개입 방안:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "아동·청소년 상담 데이터를 분석하여 위기 단계를 판단하고 상담 방향을 제시하는 전문가입니다."}, {"role": "user", "content": text}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"상담 분석 오류: {str(e)}")

    def test_aspect_sentiment(self, text: str) -> Dict[str, Any]:
        """문장 유형 및 감정 분석 테스트"""
        try:
            print("\n=== 속성기반 감정분석 테스트 ===")
            prompt = f"""다음 문장의 유형과 감정을 분석해주세요:

문장: {text}

다음 형식으로 응답해주세요:
- 문장 유형: [서술/질문/명령/감탄]
- 시제: [현재/과거/미래]
- 확실성: [높음/중간/낮음]
- 감정 극성: [긍정/중립/부정]"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "주어진 텍스트에서 자아, 가족, 친구, 학교, 미래에 대한 감정을 분석하는 전문가입니다."}, {"role": "user", "content": text}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"감정분석 오류: {str(e)}")

    def test_empathy_response(self, dialog: str) -> Dict[str, Any]:
        """공감 응답 생성 테스트"""
        try:
            print("\n=== 공감 응답 생성 테스트 ===")
            prompt = f"""다음 대화에 대해 공감적인 응답을 생성해주세요:

대화: {dialog}

다음 형식으로 응답해주세요:
- 감지된 감정:
- 공감 유형: [감정 인정/경험 공유/지지와 격려/관점 제시]
- 공감 응답:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "아동·청소년의 고민에 공감하고 적절한 상담 응답을 제공하는 전문가입니다."}, {"role": "user", "content": dialog}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"공감 응답 오류: {str(e)}")

    def test_sentence_type_analysis(self, text: str) -> Dict[str, Any]:
        """속성기반 감정분석 테스트"""
        try:
            print("\n=== 문장 유형 분석 테스트 ===")
            prompt = f"""다음 문장에 대해 속성별 감정을 분석해주세요:

문장: {text}

다음 형식으로 응답해주세요:
- 주요 속성: [자신/가족/친구/학교/미래] 중 해당되는 것
- 감정 상태:
- 감정 강도: [강/중/약]
- 원인 분석:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "주어진 문장의 유형(추론, 예측, 판단 등)을 분석하는 전문가입니다."}, {"role": "user", "content": text}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"문장 유형 분석 오류: {str(e)}")

def main():
    """테스트 실행"""
    try:
        tester = GPTTester()
        
        # 테스트 데이터
        test_data = {
            "counseling": "요즘 학교에 가기 싫어요. 친구들이 저를 무시하는 것 같고, 성적도 떨어져서 부모님한테 혼날까봐 불안해요.",
            "aspect_sentiment": "시험 성적이 떨어져서 부모님이 실망하실 것 같아 걱정돼요.",
            "empathy_dialog": "상담자: 요즘 어떻게 지내? / 내담자: 그냥 모든 게 힘들어요. 아무것도 하기 싫고...",
            "sentence_type": "친구들이랑 놀 때만 즐겁고 행복한 것 같아요."
        }

        # 각 유형별 테스트 실행
        tester.test_counseling_analysis(test_data["counseling"])
        tester.test_aspect_sentiment(test_data["aspect_sentiment"])
        tester.test_empathy_response(test_data["empathy_dialog"])
        tester.test_sentence_type_analysis(test_data["sentence_type"])

    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 