import json
import pandas as pd
from typing import Dict, List, Any

class DataPreprocessor:
    def __init__(self, config_path: str):
        """
        데이터 전처리기 초기화

        Args:
            config_path: 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def load_aspect_sentiment_data(self, file_path: str) -> pd.DataFrame:
        """문장 유형 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터프레임으로 변환
        df = pd.DataFrame([
            {
                "id": item["id"],
                "text": item["text"],
                "sentence_type": item["label"],
                "certainty": item["확실성"],
                "tense": item["시제"],
                "polarity": item["극성"]
            }
            for item in data
        ])

        return df

    def load_counseling_data(self, file_path: str) -> pd.DataFrame:
        """아동·청소년 상담 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터프레임으로 변환
        df = pd.DataFrame([
            {
                "id": item["id"],
                "text": item["text"],
                "crisis_level": item["위기_단계"],
                "emotion": item["감정"],
                "topic": item["주제"],
                "age": item["연령"],
                "gender": item["성별"]
            }
            for item in data
        ])

        return df

    def load_empathy_dialog_data(self, file_path: str) -> pd.DataFrame:
        """공감형 대화 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터프레임으로 변환
        df = pd.DataFrame([
            {
                "id": item["id"],
                "input": item["입력"],
                "response": item["응답"],
                "empathy_type": item["공감_유형"],
                "empathy_level": item["공감_수준"],
                "context": item["맥락"]
            }
            for item in data
        ])

        return df

    def load_sentence_types_data(self, file_path: str) -> pd.DataFrame:
        """속성기반 감정분석 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터프레임으로 변환
        df = pd.DataFrame([
            {
                "id": item["id"],
                "text": item["text"],
                "aspect": item["속성"],
                "sentiment": item["감정"],
                "intensity": item["강도"],
                "cause": item["원인"]
            }
            for item in data
        ])

        return df

    def preprocess(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        데이터 전처리

        Args:
            data: 원본 데이터
            data_type: 데이터 유형 (aspect_sentiment, counseling, empathy_dialog, sentence_types)

        Returns:
            전처리된 데이터
        """
        processed_data = data.copy()

        # 공통 전처리
        if 'text' in processed_data.columns:
            # 텍스트 정규화
            processed_data['text'] = processed_data['text'].str.strip()
            processed_data['text'] = processed_data['text'].str.replace(r'\s+', ' ')
            
            # 결측치 처리
            processed_data['text'] = processed_data['text'].fillna('')

        # 데이터 타입별 전처리
        if data_type == "aspect_sentiment":
            # 레이블 인코딩
            processed_data['sentence_type'] = pd.Categorical(processed_data['sentence_type']).codes
            processed_data['certainty'] = pd.Categorical(processed_data['certainty']).codes
            processed_data['tense'] = pd.Categorical(processed_data['tense']).codes
            processed_data['polarity'] = pd.Categorical(processed_data['polarity']).codes

        elif data_type == "counseling":
            # 위기 단계 인코딩
            processed_data['crisis_level'] = pd.Categorical(processed_data['crisis_level']).codes
            # 감정 레이블 인코딩
            processed_data['emotion'] = pd.Categorical(processed_data['emotion']).codes
            # 연령 정규화
            processed_data['age'] = pd.to_numeric(processed_data['age'], errors='coerce')
            processed_data['age'] = processed_data['age'].fillna(processed_data['age'].mean())

        elif data_type == "empathy_dialog":
            # 공감 유형 인코딩
            processed_data['empathy_type'] = pd.Categorical(processed_data['empathy_type']).codes
            # 공감 수준 정규화
            processed_data['empathy_level'] = pd.to_numeric(processed_data['empathy_level'], errors='coerce')
            processed_data['empathy_level'] = processed_data['empathy_level'].fillna(0)

        elif data_type == "sentence_types":
            # 감정 레이블 인코딩
            processed_data['sentiment'] = pd.Categorical(processed_data['sentiment']).codes
            # 강도 정규화
            processed_data['intensity'] = pd.to_numeric(processed_data['intensity'], errors='coerce')
            processed_data['intensity'] = processed_data['intensity'].fillna(0)

        return processed_data 