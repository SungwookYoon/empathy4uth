#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import torch
from transformers import BertTokenizer
import json

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gaslighting_detector import BertLstmHybridModel, GaslightingDetector, InterventionSystem
from src.utils.gaslighting_utils import detect_gaslighting_pattern, preprocess_conversation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gaslighting_detector_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, device, simplified=False):
    """모델 로드 함수"""
    logger.info(f"모델 로드 중: {model_path}")
    
    if simplified:
        # 간소화된 BERT-LSTM 모델
        model = BertLstmHybridModel(
            bert_model_name="klue/bert-base",
            hidden_size=768,
            lstm_hidden_size=256,
            num_classes=2,
            dropout_rate=0.2,
            num_emotions=7,
            emotion_embedding_dim=64,
            num_attention_heads=8
        ).to(device)
    else:
        # 전체 가스라이팅 탐지 시스템
        model = GaslightingDetector(
            bert_model_name="klue/bert-base",
            hidden_size=768,
            lstm_hidden_size=256,
            num_classes=2,
            dropout_rate=0.2,
            num_emotions=7,
            emotion_embedding_dim=64,
            num_attention_heads=8,
            max_conversation_turns=10,
            max_seq_length=128
        ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def demo_conversation(model, intervention_system, tokenizer, device):
    """대화형 데모 함수"""
    print("\n===== 가스라이팅 탐지 시스템 데모 =====")
    print("대화를 입력하세요. 각 턴은 새 줄로 구분하고, 빈 줄을 입력하면 분석이 시작됩니다.")
    print("'q' 또는 'quit'를 입력하면 종료합니다.")
    
    while True:
        print("\n새 대화 입력 (빈 줄로 종료):")
        conversation = []
        
        while True:
            line = input("> ")
            if not line:
                break
            if line.lower() in ['q', 'quit', 'exit']:
                return
            conversation.append(line)
        
        if not conversation:
            continue
        
        # 대화 분석
        result = detect_gaslighting_pattern(model, conversation, tokenizer=tokenizer, device=device)
        
        # 개입 전략 결정
        intervention = intervention_system.determine_intervention(
            result['gaslighting_probability'],
            result['risk_score']
        )
        
        # 결과 출력
        print("\n===== 분석 결과 =====")
        print(f"가스라이팅 탐지: {'예' if result['is_gaslighting'] else '아니오'}")
        print(f"가스라이팅 확률: {result['gaslighting_probability']:.4f}")
        print(f"위험 점수: {result['risk_score']:.4f}")
        print(f"위험 수준: {result['risk_level']}")
        
        # 개입 전략 출력 (있는 경우)
        if intervention:
            print("\n===== 개입 전략 =====")
            print(f"알림: {intervention['alert_message']}")
            print(f"설명: {intervention['explanation']}")
            if 'type_explanation' in intervention and intervention['type_explanation']:
                print(f"유형: {intervention['type_explanation']}")
            print("행동 옵션:")
            for i, action in enumerate(intervention['action_options'], 1):
                print(f"  {i}. {action}")

def analyze_file(model, intervention_system, tokenizer, file_path, output_path, device):
    """파일 분석 함수"""
    logger.info(f"파일 분석 중: {file_path}")
    
    # 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # 각 대화 분석
    for i, item in enumerate(data):
        conversation = item.get('conversation', [])
        emotion_sequence = item.get('emotions', None)
        
        if not conversation:
            continue
        
        # 대화 분석
        result = detect_gaslighting_pattern(
            model, 
            conversation, 
            emotion_sequence=emotion_sequence,
            tokenizer=tokenizer, 
            device=device
        )
        
        # 개입 전략 결정
        intervention = intervention_system.determine_intervention(
            result['gaslighting_probability'],
            result['risk_score']
        )
        
        # 결과 저장
        results.append({
            'conversation_id': item.get('id', i),
            'conversation': conversation,
            'emotions': emotion_sequence,
            'detection_result': {
                'is_gaslighting': result['is_gaslighting'],
                'gaslighting_probability': result['gaslighting_probability'],
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level']
            },
            'intervention': intervention
        })
        
        # 진행 상황 로깅
        if (i + 1) % 10 == 0:
            logger.info(f"진행 상황: {i + 1}/{len(data)} 대화 분석 완료")
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"분석 결과가 {output_path}에 저장되었습니다.")
    
    # 요약 통계
    gaslighting_count = sum(1 for r in results if r['detection_result']['is_gaslighting'])
    high_risk_count = sum(1 for r in results if r['detection_result']['risk_level'] == 'high_risk')
    medium_risk_count = sum(1 for r in results if r['detection_result']['risk_level'] == 'medium_risk')
    low_risk_count = sum(1 for r in results if r['detection_result']['risk_level'] == 'low_risk')
    
    print("\n===== 분석 요약 =====")
    print(f"총 대화 수: {len(results)}")
    print(f"가스라이팅 탐지 수: {gaslighting_count} ({gaslighting_count / len(results) * 100:.1f}%)")
    print(f"위험 수준 분포:")
    print(f"  - 높음: {high_risk_count} ({high_risk_count / len(results) * 100:.1f}%)")
    print(f"  - 중간: {medium_risk_count} ({medium_risk_count / len(results) * 100:.1f}%)")
    print(f"  - 낮음: {low_risk_count} ({low_risk_count / len(results) * 100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='가스라이팅 탐지 시스템 데모')
    
    parser.add_argument('--model_path', type=str, default='models/gaslighting_detector',
                        help='모델 파일 경로')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='실행 디바이스')
    parser.add_argument('--simplified', action='store_true',
                        help='간소화된 모델 사용 여부')
    parser.add_argument('--interactive', action='store_true',
                        help='대화형 모드 실행 여부')
    parser.add_argument('--input_file', type=str, default=None,
                        help='분석할 대화 파일 경로')
    parser.add_argument('--output_file', type=str, default='results/analysis_results.json',
                        help='분석 결과 저장 경로')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    # 모델 로드
    model = load_model(args.model_path, args.device, args.simplified)
    
    # 개입 시스템 초기화
    intervention_system = InterventionSystem()
    
    # 실행 모드 결정
    if args.interactive:
        # 대화형 모드
        demo_conversation(model, intervention_system, tokenizer, args.device)
    elif args.input_file:
        # 파일 분석 모드
        analyze_file(model, intervention_system, tokenizer, args.input_file, args.output_file, args.device)
    else:
        # 기본: 대화형 모드
        demo_conversation(model, intervention_system, tokenizer, args.device)

if __name__ == '__main__':
    main() 