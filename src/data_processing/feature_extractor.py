import pandas as pd
import numpy as np
from typing import Dict, List, Any
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        """특성 추출기 초기화"""
        self.mecab = Mecab()
        self.tfidf = TfidfVectorizer()

    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """
        언어적 특성 추출

        Args:
            text: 입력 텍스트

        Returns:
            추출된 언어적 특성
        """
        # 형태소 분석
        morphs = self.mecab.pos(text)

        # 품사별 비율 계산
        total_morphs = len(morphs) if morphs else 1
        features = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "noun_ratio": len([m for m in morphs if m[1].startswith('N')]) / total_morphs,
            "verb_ratio": len([m for m in morphs if m[1].startswith('V')]) / total_morphs,
            "adjective_ratio": len([m for m in morphs if m[1].startswith('VA')]) / total_morphs,
            "adverb_ratio": len([m for m in morphs if m[1].startswith('MA')]) / total_morphs,
            "exclamation_ratio": len([m for m in morphs if m[1] == 'IC']) / total_morphs,
            "punctuation_ratio": len([m for m in morphs if m[1] == 'SF' or m[1] == 'SP' or m[1] == 'SE']) / total_morphs,
            "average_word_length": np.mean([len(m[0]) for m in morphs]) if morphs else 0,
            "sentence_count": len([m for m in morphs if m[1] == 'SF']) + 1
        }

        return features

    def extract_sentence_type_features(self, text: str) -> Dict[str, Any]:
        """
        문장 유형 특성 추출

        Args:
            text: 입력 텍스트

        Returns:
            문장 유형 특성
        """
        # 문장 끝 패턴 분석
        endings = {
            "declarative": len([s for s in text.split('.') if s.strip().endswith('다')]),
            "interrogative": len([s for s in text.split('.') if s.strip().endswith('까') or s.strip().endswith('니')]),
            "imperative": len([s for s in text.split('.') if s.strip().endswith('라') or s.strip().endswith('요')]),
            "exclamatory": len([s for s in text.split('.') if s.strip().endswith('!')]),
        }

        # 문장 유형별 비율 계산
        total_sentences = sum(endings.values()) or 1
        features = {
            f"{key}_ratio": value / total_sentences
            for key, value in endings.items()
        }

        return features

    def extract_neologism_features(self, text: str, neologism_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        신조어 특성 추출

        Args:
            text: 입력 텍스트
            neologism_dict: 신조어 사전 (신조어: 의미)

        Returns:
            신조어 특성
        """
        # 신조어 출현 횟수
        neologism_counts = {
            word: text.count(word)
            for word in neologism_dict.keys()
        }

        # 신조어 비율
        total_words = len(text.split()) or 1
        features = {
            "neologism_ratio": sum(neologism_counts.values()) / total_words,
            "unique_neologism_count": len([c for c in neologism_counts.values() if c > 0]),
            "neologism_counts": neologism_counts
        }

        return features

    def extract_empathy_features(self, text: str) -> Dict[str, Any]:
        """
        공감 특성 추출

        Args:
            text: 입력 텍스트

        Returns:
            공감 특성
        """
        # 공감 관련 키워드
        empathy_keywords = {
            "understanding": ["이해", "알겠", "그렇군요", "그러셨군요"],
            "support": ["힘내", "응원", "할 수 있", "잘 하고 있"],
            "comfort": ["괜찮", "걱정", "마음이", "위로"],
            "agreement": ["맞아요", "동의", "그래요", "네"]
        }

        # 키워드 출현 횟수
        keyword_counts = {
            category: sum(text.count(keyword) for keyword in keywords)
            for category, keywords in empathy_keywords.items()
        }

        # 전체 공감 표현 비율
        total_words = len(text.split()) or 1
        features = {
            f"{category}_ratio": count / total_words
            for category, count in keyword_counts.items()
        }
        features["total_empathy_ratio"] = sum(keyword_counts.values()) / total_words

        return features

    def extract_aspect_sentiment_features(self, text: str) -> Dict[str, Any]:
        """
        속성별 감정 특성 추출

        Args:
            text: 입력 텍스트

        Returns:
            속성별 감정 특성
        """
        # 감정 관련 키워드
        sentiment_keywords = {
            "positive": ["좋", "행복", "기쁘", "즐겁", "감사"],
            "negative": ["나쁘", "슬프", "화나", "힘들", "싫"],
            "neutral": ["보통", "그냥", "괜찮"]
        }

        # 속성 관련 키워드
        aspect_keywords = {
            "self": ["나", "내", "저", "제"],
            "family": ["부모", "엄마", "아빠", "가족"],
            "friend": ["친구", "동료", "애들"],
            "school": ["학교", "공부", "성적", "시험"],
            "future": ["미래", "장래", "꿈", "희망"]
        }

        # 감정 키워드 출현 횟수
        sentiment_counts = {
            category: sum(text.count(keyword) for keyword in keywords)
            for category, keywords in sentiment_keywords.items()
        }

        # 속성 키워드 출현 횟수
        aspect_counts = {
            category: sum(text.count(keyword) for keyword in keywords)
            for category, keywords in aspect_keywords.items()
        }

        # 전체 단어 수 대비 비율 계산
        total_words = len(text.split()) or 1
        features = {
            f"sentiment_{category}_ratio": count / total_words
            for category, count in sentiment_counts.items()
        }
        features.update({
            f"aspect_{category}_ratio": count / total_words
            for category, count in aspect_counts.items()
        })

        # 감정 극성 점수 계산
        features["sentiment_polarity"] = (
            sentiment_counts["positive"] - sentiment_counts["negative"]
        ) / (sum(sentiment_counts.values()) or 1)

        return features

    def integrate_features(self, text: str, neologism_dict: Dict[str, str] = None) -> Dict[str, Any]:
        """
        통합 특성 추출

        Args:
            text: 입력 텍스트
            neologism_dict: 신조어 사전 (선택사항)

        Returns:
            추출된 통합 특성
        """
        # 기본 언어적 특성
        features = self.extract_linguistic_features(text)

        # 문장 유형 특성
        sentence_type_features = self.extract_sentence_type_features(text)
        features.update(sentence_type_features)

        # 신조어 특성 (사전이 제공된 경우)
        if neologism_dict:
            neologism_features = self.extract_neologism_features(text, neologism_dict)
            features.update(neologism_features)

        # 공감 특성
        empathy_features = self.extract_empathy_features(text)
        features.update(empathy_features)

        # 속성별 감정 특성
        aspect_sentiment_features = self.extract_aspect_sentiment_features(text)
        features.update(aspect_sentiment_features)

        return features 