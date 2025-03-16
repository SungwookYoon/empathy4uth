import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 각 모듈 임포트
from src.models.integrated_model import EmPathIntegratedModel
from src.models.base_models import (
    CrisisClassifier, 
    SentenceTypeClassifier, 
    AspectSentimentClassifier, 
    EmpathyResponseGenerator
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/integrated_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_test_samples(sample_dir):
    """테스트용 샘플 데이터 로드"""
    logger.info("테스트 샘플 로드 중...")
    
    test_samples = []
    
    # 상담 데이터 샘플 로드
    counseling_path = os.path.join(sample_dir, "counseling_samples.csv")
    if os.path.exists(counseling_path):
        df = pd.read_csv(counseling_path)
        for _, row in df.iterrows():
            test_samples.append({
                "text": row["text"],
                "crisis_level": row["crisis_level"],
                "source": "counseling"
            })
    
    # 문장 유형 판단 데이터 샘플 로드
    sentence_types_path = os.path.join(sample_dir, "sentence_types_samples.csv")
    if os.path.exists(sentence_types_path):
        df = pd.read_csv(sentence_types_path)
        for _, row in df.iterrows():
            test_samples.append({
                "text": row["text"],
                "sentence_type": row["sentence_type"],
                "source": "sentence_types"
            })
    
    # 속성 감정 분석 데이터 샘플 로드
    aspect_sentiment_path = os.path.join(sample_dir, "aspect_sentiment_samples.csv")
    if os.path.exists(aspect_sentiment_path):
        df = pd.read_csv(aspect_sentiment_path)
        for _, row in df.iterrows():
            test_samples.append({
                "text": row["text"],
                "aspect": row["aspect"],
                "sentiment": row["sentiment"],
                "source": "aspect_sentiment"
            })
    
    # 공감형 대화 데이터 샘플 로드
    empathy_dialog_path = os.path.join(sample_dir, "empathy_dialog_samples.csv")
    if os.path.exists(empathy_dialog_path):
        df = pd.read_csv(empathy_dialog_path)
        for _, row in df.iterrows():
            test_samples.append({
                "text": row["query"],
                "response": row["response"],
                "empathy_type": row["empathy_type"],
                "source": "empathy_dialog"
            })
    
    logger.info(f"총 {len(test_samples)}개의 테스트 샘플을 로드했습니다.")
    return test_samples

def evaluate_model(model, test_samples):
    """통합 모델 평가"""
    logger.info("통합 모델 평가 중...")
    
    # 평가 지표 초기화
    metrics = {
        "crisis_accuracy": 0,
        "sentence_type_accuracy": 0,
        "aspect_accuracy": 0,
        "sentiment_accuracy": 0,
        "empathy_type_accuracy": 0,
        "total_samples": len(test_samples),
        "crisis_samples": 0,
        "sentence_type_samples": 0,
        "aspect_sentiment_samples": 0,
        "empathy_samples": 0
    }
    
    results = []
    
    for sample in tqdm(test_samples, desc="샘플 평가 중"):
        text = sample["text"]
        source = sample["source"]
        
        # 통합 분석 수행
        try:
            analysis = model.analyze(text)
            
            # 결과 저장
            result = {
                "text": text,
                "source": source,
                "analysis": analysis
            }
            
            # 정답과 비교하여 정확도 계산
            if source == "counseling" and "crisis_level" in sample:
                metrics["crisis_samples"] += 1
                pred_crisis = analysis["crisis_analysis"]["final_crisis_level"]
                if pred_crisis == sample["crisis_level"]:
                    metrics["crisis_accuracy"] += 1
            
            elif source == "sentence_types" and "sentence_type" in sample:
                metrics["sentence_type_samples"] += 1
                # 가장 많이 나온 문장 유형 선택
                type_counts = analysis["type_analysis"]["type_counts"]
                pred_type = max(type_counts, key=type_counts.get) if type_counts else None
                if pred_type == sample["sentence_type"]:
                    metrics["sentence_type_accuracy"] += 1
            
            elif source == "aspect_sentiment" and "aspect" in sample and "sentiment" in sample:
                metrics["aspect_sentiment_samples"] += 1
                # 결과에서 해당 속성과 감정을 찾음
                aspect_sentiment_matrix = analysis["aspect_sentiment_analysis"]["aspect_sentiment_matrix"]
                
                # 속성 정확도 평가
                aspect_total_counts = {}
                for aspect, sentiments in aspect_sentiment_matrix.items():
                    aspect_total_counts[aspect] = sum(sentiments.values())
                
                pred_aspect = max(aspect_total_counts, key=aspect_total_counts.get) if aspect_total_counts else None
                if pred_aspect == sample["aspect"]:
                    metrics["aspect_accuracy"] += 1
                
                # 감정 정확도 평가 (해당 속성에 대한 감정)
                if sample["aspect"] in aspect_sentiment_matrix:
                    sentiments = aspect_sentiment_matrix[sample["aspect"]]
                    pred_sentiment = max(sentiments, key=sentiments.get) if sentiments else None
                    if pred_sentiment == sample["sentiment"]:
                        metrics["sentiment_accuracy"] += 1
            
            elif source == "empathy_dialog" and "empathy_type" in sample:
                metrics["empathy_samples"] += 1
                pred_empathy_type = analysis["empathy_response"]["empathy_type"]
                if pred_empathy_type == sample["empathy_type"]:
                    metrics["empathy_type_accuracy"] += 1
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"샘플 {text[:30]}... 평가 중 오류 발생: {str(e)}")
    
    # 정확도 계산
    if metrics["crisis_samples"] > 0:
        metrics["crisis_accuracy"] /= metrics["crisis_samples"]
    if metrics["sentence_type_samples"] > 0:
        metrics["sentence_type_accuracy"] /= metrics["sentence_type_samples"]
    if metrics["aspect_sentiment_samples"] > 0:
        metrics["aspect_accuracy"] /= metrics["aspect_sentiment_samples"]
        metrics["sentiment_accuracy"] /= metrics["aspect_sentiment_samples"]
    if metrics["empathy_samples"] > 0:
        metrics["empathy_type_accuracy"] /= metrics["empathy_samples"]
    
    logger.info("평가 완료:")
    logger.info(f"- 위기 단계 분류 정확도: {metrics['crisis_accuracy']:.4f} ({metrics['crisis_samples']} 샘플)")
    logger.info(f"- 문장 유형 분류 정확도: {metrics['sentence_type_accuracy']:.4f} ({metrics['sentence_type_samples']} 샘플)")
    logger.info(f"- 속성 분류 정확도: {metrics['aspect_accuracy']:.4f} ({metrics['aspect_sentiment_samples']} 샘플)")
    logger.info(f"- 감정 분류 정확도: {metrics['sentiment_accuracy']:.4f} ({metrics['aspect_sentiment_samples']} 샘플)")
    logger.info(f"- 공감 유형 분류 정확도: {metrics['empathy_type_accuracy']:.4f} ({metrics['empathy_samples']} 샘플)")
    
    return metrics, results

def fine_tune_weights(model, test_samples, output_dir):
    """통합 모델 가중치 미세 조정"""
    logger.info("통합 모델 가중치 미세 조정 중...")
    
    # 조정할 가중치 세트
    weight_sets = [
        # 기본 가중치
        {
            "crisis_weights": model.crisis_weights.copy(),
            "aspect_weights": model.aspect_weights.copy(),
            "sentiment_weights": model.sentiment_weights.copy(),
            "sentence_type_weights": model.sentence_type_weights.copy()
        },
        # 위기 단계 가중치 강화
        {
            "crisis_weights": {
                "정상군": 0.1,
                "관찰필요": 0.3,
                "상담필요": 0.7,
                "학대의심": 0.9,
                "응급": 1.0
            },
            "aspect_weights": model.aspect_weights.copy(),
            "sentiment_weights": model.sentiment_weights.copy(),
            "sentence_type_weights": model.sentence_type_weights.copy()
        },
        # 감정 가중치 강화
        {
            "crisis_weights": model.crisis_weights.copy(),
            "aspect_weights": model.aspect_weights.copy(),
            "sentiment_weights": {
                "긍정": 0.2,
                "중립": 0.4,
                "부정": 1.0
            },
            "sentence_type_weights": model.sentence_type_weights.copy()
        },
        # 문장 유형 가중치 강화
        {
            "crisis_weights": model.crisis_weights.copy(),
            "aspect_weights": model.aspect_weights.copy(),
            "sentiment_weights": model.sentiment_weights.copy(),
            "sentence_type_weights": {
                "사실진술": 0.3,
                "감정표현": 1.0,
                "의견제시": 0.5,
                "도움요청": 1.0,
                "기타": 0.2
            }
        }
    ]
    
    best_metrics = None
    best_weights = None
    best_score = -1
    
    for i, weights in enumerate(weight_sets):
        logger.info(f"가중치 세트 {i+1}/{len(weight_sets)} 평가 중...")
        
        # 가중치 설정
        model.crisis_weights = weights["crisis_weights"]
        model.aspect_weights = weights["aspect_weights"]
        model.sentiment_weights = weights["sentiment_weights"]
        model.sentence_type_weights = weights["sentence_type_weights"]
        
        # 평가
        metrics, _ = evaluate_model(model, test_samples)
        
        # 종합 점수 계산 (각 정확도의 가중 평균)
        score = (
            0.3 * metrics["crisis_accuracy"] +
            0.2 * metrics["sentence_type_accuracy"] +
            0.2 * metrics["aspect_accuracy"] +
            0.2 * metrics["sentiment_accuracy"] +
            0.1 * metrics["empathy_type_accuracy"]
        )
        
        logger.info(f"가중치 세트 {i+1} 종합 점수: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_weights = weights
    
    # 최적의 가중치 적용 및 저장
    if best_weights:
        model.crisis_weights = best_weights["crisis_weights"]
        model.aspect_weights = best_weights["aspect_weights"]
        model.sentiment_weights = best_weights["sentiment_weights"]
        model.sentence_type_weights = best_weights["sentence_type_weights"]
        
        # 가중치 저장
        weights_path = os.path.join(output_dir, "optimal_weights.json")
        with open(weights_path, "w") as f:
            json.dump(best_weights, f, indent=4)
        
        logger.info(f"최적의 가중치가 {weights_path}에 저장되었습니다.")
        logger.info(f"최적의 종합 점수: {best_score:.4f}")
    
    return best_metrics

def test_examples(model):
    """예제 텍스트로 모델 테스트"""
    logger.info("예제 텍스트로 모델 테스트 중...")
    
    test_texts = [
        "요즘 학교에서 친구들이 자꾸 나를 놀려요. 집에 가기가 싫어요.",
        "부모님이 자꾸 싸우셔서 집에 가기가 무서워요. 어떨 때는 소리를 너무 지르셔서 이불 속에 숨어있어요.",
        "시험을 잘 봐서 기분이 좋아요. 부모님도 많이 칭찬해주셨어요.",
        "친구가 자기 죽고 싶다고 했어요. 어떻게 해야 할지 모르겠어요. 도와주세요."
    ]
    
    for i, text in enumerate(test_texts):
        logger.info(f"테스트 예제 {i+1}/{len(test_texts)}: {text}")
        
        try:
            logger.info("분석 시작...")
            analysis = model.analyze(text)
            logger.info("분석 완료!")
            
            # 주요 결과 로깅
            logger.info(f"- 위기 수준: {analysis['crisis_analysis']['final_crisis_level']}")
            logger.info(f"- 위험 점수: {analysis['risk_score']:.4f}")
            logger.info(f"- 공감 응답: {analysis['empathy_response']['response']}")
            
            # 속성별 감정 분석 결과
            logger.info("- 속성별 감정:")
            for aspect, sentiments in analysis["aspect_sentiment_analysis"]["aspect_sentiment_matrix"].items():
                if sum(sentiments.values()) > 0:
                    main_sentiment = max(sentiments, key=sentiments.get)
                    logger.info(f"  * {aspect}: {main_sentiment} ({sentiments[main_sentiment]}건)")
            
            logger.info("")  # 빈 줄 추가
            
        except Exception as e:
            logger.error(f"예제 분석 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return True

def main():
    parser = argparse.ArgumentParser(description="EmPath 통합 모델 훈련 및 평가")
    parser.add_argument("--base_models_dir", type=str, default="models", help="기본 모델 디렉토리")
    parser.add_argument("--output_dir", type=str, default="models/integrated_model", help="통합 모델 저장 디렉토리")
    parser.add_argument("--sample_dir", type=str, default="data/samples", help="테스트 샘플 디렉토리")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="학습 디바이스")
    parser.add_argument("--epochs", type=int, default=10, help="통합 모델 훈련 에포크 수")
    parser.add_argument("--test_examples", action="store_true", help="테스트 예시 분석 수행 여부")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("EmPath 통합 모델 훈련/평가 시작...")
    
    # 통합 모델 인스턴스 생성
    model = EmPathIntegratedModel(
        model_dir=args.base_models_dir,
        model_name="klue/bert-base",
        device=args.device
    )
    
    # 예제 텍스트로 테스트
    if args.test_examples:
        test_examples(model)
    
    # 모델 평가
    # 테스트 샘플 로드
    test_samples = load_test_samples(args.sample_dir)
    
    # 모델 평가
    metrics, results = evaluate_model(model, test_samples)
    
    # 평가 결과 저장
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 상세 결과 저장 (선택적)
    # results_path = os.path.join(args.output_dir, "evaluation_results.json")
    # with open(results_path, "w") as f:
    #     json.dump(results, f, indent=4)
    
    logger.info(f"평가 결과가 {metrics_path}에 저장되었습니다.")
    
    # 가중치 미세 조정
    fine_tune_weights(model, test_samples, args.output_dir)
    
    logger.info("EmPath 통합 모델 훈련/평가 완료!")

if __name__ == "__main__":
    main() 