import torch
import torch.nn as nn
import os
import json
import numpy as np
from transformers import BertTokenizer

from src.models.base_models import (
    CrisisClassifier, 
    SentenceTypeClassifier, 
    AspectSentimentClassifier, 
    EmpathyResponseGenerator
)

class EmPathIntegratedModel:
    """EmPath 통합 모델 클래스"""
    def __init__(
        self, 
        model_dir="models/base_models", 
        model_name="klue/bert-base",
        device="cuda"
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 각 모델 로드
        self.crisis_model = self._load_crisis_model()
        self.sentence_type_model = self._load_sentence_type_model()
        self.aspect_sentiment_model = self._load_aspect_sentiment_model()
        self.empathy_model = self._load_empathy_model()
        
        # 문장 단위 분석을 위한 설정
        self.max_sentence_length = 128
        
        # 위기 수준별 가중치 (위기 수준이 높을수록 더 큰 가중치)
        self.crisis_weights = {
            "정상군": 0.2,
            "관찰필요": 0.4,
            "상담필요": 0.6,
            "학대의심": 0.8,
            "응급": 1.0
        }
        
        # 속성별 가중치 (모든 속성에 동일한 가중치 부여)
        self.aspect_weights = {
            "자아": 1.0,
            "가족": 1.0,
            "친구": 1.0,
            "학교": 1.0,
            "미래": 1.0
        }
        
        # 감정별 가중치 (부정적 감정에 더 큰 가중치)
        self.sentiment_weights = {
            "긍정": 0.3,
            "중립": 0.5,
            "부정": 1.0
        }
        
        # 문장 유형별 가중치 (감정 표현과 도움 요청에 더 큰 가중치)
        self.sentence_type_weights = {
            "사실진술": 0.4,
            "감정표현": 1.0,
            "의견제시": 0.6,
            "도움요청": 0.9,
            "기타": 0.3
        }
        
    def _load_crisis_model(self):
        """위기 단계 분류 모델 로드"""
        model_path = os.path.join(self.model_dir, "crisis_model", "model.pt")
        mapping_path = os.path.join(self.model_dir, "crisis_model", "label_mapping.json")
        
        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            print(f"위기 단계 분류 모델 또는 매핑 파일을 찾을 수 없습니다.")
            return None
        
        # 매핑 로드
        with open(mapping_path, "r") as f:
            label_mapping = json.load(f)
        
        # 역매핑 생성
        inv_label_mapping = {str(v): k for k, v in label_mapping.items()}
        
        # 디버깅을 위한 매핑 출력
        print("Crisis Label Mapping:", label_mapping)
        print("Crisis Inv Label Mapping:", inv_label_mapping)
        
        # 클래스 수 확인
        num_classes = len(label_mapping)
        
        # 모델 초기화 및 가중치 로드
        model = CrisisClassifier(model_name=self.model_name, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return {
            "model": model,
            "label_mapping": label_mapping,
            "inv_label_mapping": inv_label_mapping
        }
    
    def _load_sentence_type_model(self):
        """문장 유형 분류 모델 로드"""
        model_path = os.path.join(self.model_dir, "sentence_type_model", "model.pt")
        mapping_path = os.path.join(self.model_dir, "sentence_type_model", "label_mapping.json")
        
        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            print(f"문장 유형 분류 모델 또는 매핑 파일을 찾을 수 없습니다.")
            return None
        
        # 매핑 로드
        with open(mapping_path, "r") as f:
            label_mapping = json.load(f)
        
        # 역매핑 생성
        inv_label_mapping = {str(v): k for k, v in label_mapping.items()}
        
        # 디버깅을 위한 매핑 출력
        print("Sentence Type Label Mapping:", label_mapping)
        print("Sentence Type Inv Label Mapping:", inv_label_mapping)
        
        # 클래스 수 확인
        num_classes = len(label_mapping)
        
        # 모델 초기화 및 가중치 로드
        model = SentenceTypeClassifier(model_name=self.model_name, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return {
            "model": model,
            "label_mapping": label_mapping,
            "inv_label_mapping": inv_label_mapping
        }
    
    def _load_aspect_sentiment_model(self):
        """속성 감정 분석 모델 로드"""
        model_path = os.path.join(self.model_dir, "aspect_sentiment_model", "model.pt")
        label_mapping_path = os.path.join(self.model_dir, "aspect_sentiment_model", "label_mapping.json")
        
        if not os.path.exists(model_path) or not os.path.exists(label_mapping_path):
            print(f"속성 감정 분석 모델 또는 매핑 파일을 찾을 수 없습니다.")
            return None
        
        # 매핑 로드
        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)
            
        aspect_mapping = label_mapping["aspect"]
        sentiment_mapping = label_mapping["sentiment"]
        
        # 역매핑 생성
        inv_aspect_mapping = {str(v): k for k, v in aspect_mapping.items()}
        inv_sentiment_mapping = {str(v): k for k, v in sentiment_mapping.items()}
        
        # 디버깅을 위한 매핑 출력
        print("Aspect Mapping:", aspect_mapping)
        print("Sentiment Mapping:", sentiment_mapping)
        print("Inv Aspect Mapping:", inv_aspect_mapping)
        print("Inv Sentiment Mapping:", inv_sentiment_mapping)
        
        # 클래스 수 확인
        num_aspects = len(aspect_mapping)
        num_sentiments = len(sentiment_mapping)
        
        # 모델 초기화 및 가중치 로드
        model = AspectSentimentClassifier(
            model_name=self.model_name, 
            num_aspects=num_aspects, 
            num_sentiments=num_sentiments
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return {
            "model": model,
            "aspect_mapping": aspect_mapping,
            "sentiment_mapping": sentiment_mapping,
            "inv_aspect_mapping": inv_aspect_mapping,
            "inv_sentiment_mapping": inv_sentiment_mapping
        }
    
    def _load_empathy_model(self):
        """공감 응답 생성 모델 로드"""
        model_path = os.path.join(self.model_dir, "empathy_model", "model.pt")
        mapping_path = os.path.join(self.model_dir, "empathy_model", "empathy_mapping.json")
        
        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            print(f"공감 응답 생성 모델 또는 매핑 파일을 찾을 수 없습니다.")
            return None
        
        # 매핑 로드
        with open(mapping_path, "r") as f:
            empathy_mapping = json.load(f)
        
        # 모델 초기화 및 가중치 로드
        model = EmpathyResponseGenerator(model_name=self.model_name)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        # 역 매핑 생성 (예측 결과 해석용)
        inv_empathy_mapping = {v: k for k, v in empathy_mapping.items()}
        
        return {
            "model": model,
            "empathy_mapping": empathy_mapping,
            "inv_empathy_mapping": inv_empathy_mapping
        }
    
    def _preprocess_text(self, text):
        """텍스트 전처리 및 문장 분리"""
        # 기본적인 전처리 (실제로는 더 복잡한 전처리가 필요할 수 있음)
        text = text.strip()
        
        # 간단한 문장 분리 (실제로는 더 정교한 방법 필요)
        sentences = []
        for line in text.split('\n'):
            if line.strip():
                # 마침표, 물음표, 느낌표로 문장 분리
                for sent in line.replace('!', '.').replace('?', '.').split('.'):
                    if sent.strip():
                        sentences.append(sent.strip())
        
        return sentences
    
    def _tokenize(self, text):
        """텍스트 토큰화"""
        return self.tokenizer(
            text,
            max_length=self.max_sentence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def analyze_crisis_level(self, text):
        """위기 단계 분석"""
        if self.crisis_model is None:
            return {"error": "위기 단계 분류 모델이 로드되지 않았습니다."}
        
        sentences = self._preprocess_text(text)
        if not sentences:
            return {"error": "분석할 문장이 없습니다."}
        
        results = []
        crisis_counts = {label: 0 for label in self.crisis_weights.keys()}
        
        for sentence in sentences:
            encoding = self._tokenize(sentence)
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding["token_type_ids"].to(self.device) if "token_type_ids" in encoding else None
            
            with torch.no_grad():
                outputs = self.crisis_model["model"](input_ids, attention_mask, token_type_ids)
                pred_label = torch.argmax(outputs, dim=1).item()
                pred_label_str = str(int(pred_label))
                
                # 키 오류 처리
                if pred_label_str in self.crisis_model["inv_label_mapping"]:
                    pred_crisis = self.crisis_model["inv_label_mapping"][pred_label_str]
                else:
                    print(f"경고: 키 '{pred_label_str}'가 inv_label_mapping에 없습니다. 대체값 사용.")
                    # 기본값으로 첫 번째 레이블 사용
                    pred_crisis = list(self.crisis_weights.keys())[0]
                
                # 결과 저장
                results.append({
                    "sentence": sentence,
                    "crisis_level": pred_crisis,
                    "confidence": torch.softmax(outputs, dim=1)[0][pred_label].item()
                })
                
                # 위기 수준 카운트
                crisis_counts[pred_crisis] += 1
        
        # 종합 위기 수준 계산 (가중 평균)
        weighted_sum = sum(crisis_counts[level] * self.crisis_weights[level] for level in crisis_counts)
        total_sentences = len(sentences)
        crisis_score = weighted_sum / total_sentences if total_sentences > 0 else 0
        
        # 점수에 따른 최종 위기 수준 결정
        if crisis_score < 0.3:
            final_crisis = "정상군"
        elif crisis_score < 0.5:
            final_crisis = "관찰필요"
        elif crisis_score < 0.7:
            final_crisis = "상담필요"
        elif crisis_score < 0.9:
            final_crisis = "학대의심"
        else:
            final_crisis = "응급"
        
        return {
            "sentence_results": results,
            "crisis_counts": crisis_counts,
            "crisis_score": crisis_score,
            "final_crisis_level": final_crisis
        }
    
    def analyze_sentence_types(self, text):
        """문장 유형 분석"""
        if self.sentence_type_model is None:
            return {"error": "문장 유형 분류 모델이 로드되지 않았습니다."}
        
        sentences = self._preprocess_text(text)
        if not sentences:
            return {"error": "분석할 문장이 없습니다."}
        
        results = []
        type_counts = {}
        
        for sentence in sentences:
            encoding = self._tokenize(sentence)
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding["token_type_ids"].to(self.device) if "token_type_ids" in encoding else None
            
            with torch.no_grad():
                outputs = self.sentence_type_model["model"](input_ids, attention_mask, token_type_ids)
                pred_label = torch.argmax(outputs, dim=1).item()
                pred_label_str = str(int(pred_label))
                
                # 키 오류 처리
                if pred_label_str in self.sentence_type_model["inv_label_mapping"]:
                    pred_type = self.sentence_type_model["inv_label_mapping"][pred_label_str]
                else:
                    print(f"경고: 키 '{pred_label_str}'가 inv_label_mapping에 없습니다. 대체값 사용.")
                    # 기본값으로 "중립" 사용
                    pred_type = "중립"
                
                # 결과 저장
                results.append({
                    "sentence": sentence,
                    "sentence_type": pred_type,
                    "confidence": torch.softmax(outputs, dim=1)[0][pred_label].item()
                })
                
                # 문장 유형 카운트
                type_counts[pred_type] = type_counts.get(pred_type, 0) + 1
        
        return {
            "sentence_results": results,
            "type_counts": type_counts
        }
    
    def analyze_aspect_sentiment(self, text):
        """속성별 감정 분석"""
        if self.aspect_sentiment_model is None:
            return {"error": "속성 감정 분석 모델이 로드되지 않았습니다."}
        
        sentences = self._preprocess_text(text)
        if not sentences:
            return {"error": "분석할 문장이 없습니다."}
        
        results = []
        aspects = self.aspect_weights.keys()
        sentiments = ["긍정", "부정", "중립"]  # 기본 감정 클래스
        aspect_sentiment_matrix = {aspect: {sentiment: 0 for sentiment in sentiments} for aspect in aspects}
        
        for sentence in sentences:
            encoding = self._tokenize(sentence)
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding["token_type_ids"].to(self.device) if "token_type_ids" in encoding else None
            
            with torch.no_grad():
                aspect_outputs, sentiment_outputs = self.aspect_sentiment_model["model"](input_ids, attention_mask, token_type_ids)
                
                pred_aspect = torch.argmax(aspect_outputs, dim=1).item()
                pred_sentiment = torch.argmax(sentiment_outputs, dim=1).item()
                
                pred_aspect_str = str(int(pred_aspect))
                pred_sentiment_str = str(int(pred_sentiment))
                
                # 키 오류 처리
                if pred_aspect_str in self.aspect_sentiment_model["inv_aspect_mapping"]:
                    aspect = self.aspect_sentiment_model["inv_aspect_mapping"][pred_aspect_str]
                else:
                    print(f"경고: 키 '{pred_aspect_str}'가 inv_aspect_mapping에 없습니다. 대체값 사용.")
                    # 기본값으로 첫 번째 속성 사용
                    aspect = list(self.aspect_weights.keys())[0]
                
                if pred_sentiment_str in self.aspect_sentiment_model["inv_sentiment_mapping"]:
                    sentiment = self.aspect_sentiment_model["inv_sentiment_mapping"][pred_sentiment_str]
                else:
                    print(f"경고: 키 '{pred_sentiment_str}'가 inv_sentiment_mapping에 없습니다. 대체값 사용.")
                    # 기본값으로 "중립" 사용
                    sentiment = "중립"
                
                # 결과 저장
                results.append({
                    "sentence": sentence,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "aspect_confidence": torch.softmax(aspect_outputs, dim=1)[0][pred_aspect].item(),
                    "sentiment_confidence": torch.softmax(sentiment_outputs, dim=1)[0][pred_sentiment].item()
                })
                
                # 속성-감정 매트릭스 업데이트
                aspect_sentiment_matrix[aspect][sentiment] += 1
        
        return {
            "sentence_results": results,
            "aspect_sentiment_matrix": aspect_sentiment_matrix
        }
    
    def generate_empathic_response(self, text, crisis_analysis=None, type_analysis=None, sentiment_analysis=None):
        """공감적 응답 생성"""
        if self.empathy_model is None:
            return {"error": "공감 응답 생성 모델이 로드되지 않았습니다."}
        
        # 필요한 분석이 없으면 수행
        if crisis_analysis is None:
            crisis_analysis = self.analyze_crisis_level(text)
        
        if type_analysis is None:
            type_analysis = self.analyze_sentence_types(text)
        
        if sentiment_analysis is None:
            sentiment_analysis = self.analyze_aspect_sentiment(text)
        
        # 최적의 공감 유형 선택
        empathy_type = self._select_optimal_empathy_type(crisis_analysis, type_analysis, sentiment_analysis)
        
        # 간단한 템플릿 기반 응답 (실제로는 생성 모델 사용)
        if empathy_type == "조언":
            response = f"당신의 상황에 대해 생각해보았어요. 다음과 같은 방법을 시도해보는 건 어떨까요? "
        elif empathy_type == "격려":
            response = f"정말 힘든 상황이지만, 당신은 충분히 이겨낼 수 있어요. "
        elif empathy_type == "위로":
            response = f"그런 경험을 했다니 정말 마음이 아프네요. 당신의 감정은 충분히 이해할 수 있어요. "
        elif empathy_type == "동조":
            response = f"그런 상황이라면 저도 비슷하게 느꼈을 것 같아요. "
        else:
            response = f"당신의 이야기를 들어보니 "
        
        # 위기 수준에 따른 응답 추가
        crisis_level = crisis_analysis["final_crisis_level"]
        if crisis_level in ["학대의심", "응급"]:
            response += "지금 상황이 매우 위험할 수 있습니다. 전문가의 도움을 즉시 받는 것이 좋겠습니다. "
        elif crisis_level == "상담필요":
            response += "상담사와 대화를 통해 더 나은 해결책을 찾을 수 있을 것입니다. "
        
        # 부정적 감정에 대한 공감 추가
        negative_aspects = []
        for aspect, sentiments in sentiment_analysis["aspect_sentiment_matrix"].items():
            if sentiments["부정"] > 0:
                negative_aspects.append(aspect)
        
        if negative_aspects:
            aspects_str = ", ".join(negative_aspects)
            response += f"{aspects_str}에 대한 부정적인 감정이 있으신 것 같아요. 그런 감정을 느끼는 것은 자연스러운 일입니다. "
        
        return {
            "empathy_type": empathy_type,
            "response": response.strip()
        }
    
    def _select_optimal_empathy_type(self, crisis_analysis, type_analysis, sentiment_analysis):
        """최적의 공감 유형 선택"""
        # 위기 수준에 따른 가중치
        crisis_level = crisis_analysis["final_crisis_level"]
        crisis_weight = self.crisis_weights[crisis_level]
        
        # 문장 유형에 따른 가중치
        type_scores = {}
        for type_name, count in type_analysis["type_counts"].items():
            type_scores[type_name] = count * self.sentence_type_weights[type_name]
        
        # 부정적 감정의 비율 계산
        negative_count = 0
        total_count = 0
        for aspect, sentiments in sentiment_analysis["aspect_sentiment_matrix"].items():
            for sentiment, count in sentiments.items():
                total_count += count
                if sentiment == "부정":
                    negative_count += count
        
        negative_ratio = negative_count / total_count if total_count > 0 else 0
        
        # 공감 유형별 점수 계산
        empathy_scores = {
            "조언": 0,
            "격려": 0,
            "위로": 0,
            "동조": 0
        }
        
        # 1. 위기 수준이 높을수록 위로와 조언 강화
        if crisis_level in ["학대의심", "응급"]:
            empathy_scores["위로"] += crisis_weight * 2
            empathy_scores["조언"] += crisis_weight * 1.5
        elif crisis_level == "상담필요":
            empathy_scores["위로"] += crisis_weight
            empathy_scores["격려"] += crisis_weight
        else:
            empathy_scores["동조"] += (1 - crisis_weight)
        
        # 2. 문장 유형에 따른 점수
        if "감정표현" in type_scores and type_scores["감정표현"] > 0:
            empathy_scores["위로"] += type_scores["감정표현"] * 0.5
            empathy_scores["동조"] += type_scores["감정표현"] * 0.3
        
        if "도움요청" in type_scores and type_scores["도움요청"] > 0:
            empathy_scores["조언"] += type_scores["도움요청"] * 0.7
        
        if "의견제시" in type_scores and type_scores["의견제시"] > 0:
            empathy_scores["동조"] += type_scores["의견제시"] * 0.5
        
        # 3. 부정적 감정 비율에 따른 점수
        empathy_scores["위로"] += negative_ratio * 0.8
        empathy_scores["격려"] += negative_ratio * 0.6
        
        # 최고 점수의 공감 유형 선택
        return max(empathy_scores, key=empathy_scores.get)
    
    def analyze(self, text):
        """통합 분석"""
        # 각 모델 분석 수행
        crisis_analysis = self.analyze_crisis_level(text)
        type_analysis = self.analyze_sentence_types(text)
        aspect_sentiment_analysis = self.analyze_aspect_sentiment(text)
        
        # 통합 위험 점수 계산
        risk_score = self._calculate_integrated_risk_score(
            crisis_analysis, type_analysis, aspect_sentiment_analysis
        )
        
        # 공감 응답 생성
        empathy_response = self.generate_empathic_response(
            text, crisis_analysis, type_analysis, aspect_sentiment_analysis
        )
        
        return {
            "crisis_analysis": crisis_analysis,
            "type_analysis": type_analysis,
            "aspect_sentiment_analysis": aspect_sentiment_analysis,
            "risk_score": risk_score,
            "empathy_response": empathy_response
        }
    
    def _calculate_integrated_risk_score(self, crisis_analysis, type_analysis, sentiment_analysis):
        """통합 위험 점수 계산"""
        # 1. 위기 수준 점수
        crisis_level = crisis_analysis["final_crisis_level"]
        crisis_score = self.crisis_weights[crisis_level]
        
        # 2. 부정적 감정 점수
        negative_score = 0
        total_count = 0
        for aspect, sentiments in sentiment_analysis["aspect_sentiment_matrix"].items():
            for sentiment, count in sentiments.items():
                if sentiment == "부정":
                    negative_score += count * self.sentiment_weights[sentiment] * self.aspect_weights[aspect]
                total_count += count
        
        normalized_negative_score = negative_score / total_count if total_count > 0 else 0
        
        # 3. 문장 유형 점수
        help_request_score = 0
        emotional_score = 0
        sentence_count = 0
        
        for type_name, count in type_analysis["type_counts"].items():
            sentence_count += count
            if type_name == "도움요청":
                help_request_score += count * self.sentence_type_weights[type_name]
            elif type_name == "감정표현":
                emotional_score += count * self.sentence_type_weights[type_name]
        
        normalized_help_request_score = help_request_score / sentence_count if sentence_count > 0 else 0
        normalized_emotional_score = emotional_score / sentence_count if sentence_count > 0 else 0
        
        # 가중치 적용 및 정규화
        final_risk_score = (
            0.5 * crisis_score + 
            0.3 * normalized_negative_score +
            0.1 * normalized_help_request_score +
            0.1 * normalized_emotional_score
        )
        
        return min(max(final_risk_score, 0), 1)  # 0~1 범위로 제한 