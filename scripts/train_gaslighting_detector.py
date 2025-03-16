#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
import json
from tqdm import tqdm

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gaslighting_detector import BertLstmHybridModel, GaslightingDetector, InterventionSystem
from src.utils.train_utils import EarlyStopping, compute_class_weights

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gaslighting_detector_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GaslightingDataset(Dataset):
    """가스라이팅 탐지를 위한 데이터셋 클래스"""
    def __init__(self, conversations, labels, tokenizer, max_seq_length=128, max_turns=10, emotion_tags=None):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_turns = max_turns
        self.emotion_tags = emotion_tags
        
        # 감정 태그 매핑 (문자열 -> 정수)
        self.emotion_map = {
            '기쁨': 0, 
            '슬픔': 1, 
            '분노': 2, 
            '공포': 3, 
            '놀람': 4, 
            '혐오': 5, 
            '중립': 6,
            None: 6  # 감정 태그가 없는 경우 중립으로 처리
        }
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        label = self.labels[idx]
        
        # 대화 턴 처리
        input_ids_list = []
        attention_mask_list = []
        emotion_ids_list = []
        
        # 각 턴별 처리
        for i, turn in enumerate(conversation[:self.max_turns]):
            # 텍스트 인코딩
            encoding = self.tokenizer(
                turn,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_list.append(encoding['input_ids'].squeeze())
            attention_mask_list.append(encoding['attention_mask'].squeeze())
            
            # 감정 태그 처리 (있는 경우)
            if self.emotion_tags is not None and idx < len(self.emotion_tags) and i < len(self.emotion_tags[idx]):
                emotion = self.emotion_tags[idx][i]
                emotion_id = self.emotion_map.get(emotion, 6)  # 매핑에 없는 경우 중립으로 처리
            else:
                emotion_id = 6  # 감정 태그가 없는 경우 중립으로 처리
            
            emotion_ids_list.append(emotion_id)
        
        # 패딩 처리 (대화 턴이 max_turns보다 적은 경우)
        while len(input_ids_list) < self.max_turns:
            # 빈 턴 추가
            padding_encoding = self.tokenizer(
                "",
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_list.append(padding_encoding['input_ids'].squeeze())
            attention_mask_list.append(padding_encoding['attention_mask'].squeeze())
            emotion_ids_list.append(6)  # 패딩 턴의 감정은 중립으로 처리
        
        # 텐서로 변환
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        emotion_ids = torch.tensor(emotion_ids_list, dtype=torch.long)
        
        # 단일 대화 턴으로 처리 (간소화된 모델용)
        # 모든 턴을 하나의 텍스트로 연결
        combined_text = " ".join(conversation[:self.max_turns])
        combined_encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_seq_length * 2,  # 더 긴 시퀀스 허용
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        combined_input_ids = combined_encoding['input_ids'].squeeze()
        combined_attention_mask = combined_encoding['attention_mask'].squeeze()
        
        return {
            'conversation_input_ids': input_ids,
            'conversation_attention_mask': attention_mask,
            'emotion_ids': emotion_ids,
            'input_ids': combined_input_ids,  # 단일 턴 입력
            'attention_mask': combined_attention_mask,  # 단일 턴 마스크
            'label': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    """텍스트 전처리 함수"""
    if not isinstance(text, str):
        return ""
    
    # 기본 전처리
    text = text.strip()
    if not text:
        return ""
    
    # URL 제거
    text = re.sub(r'http\S+', '', text)
    
    # 이메일 주소 제거
    text = re.sub(r'\S+@\S+', '', text)
    
    # 특수문자 처리
    text = re.sub(r'[^\w\s\.,!?]', ' ', text)
    
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 문장 끝 처리
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def restore_ellipsis(current_utterance, previous_utterances, tokenizer):
    """생략된 주어/목적어 복원 함수 (간소화된 버전)"""
    # 실제 구현에서는 더 복잡한 로직이 필요할 수 있음
    if not current_utterance.strip():
        return current_utterance
    
    # 주어가 생략된 경우 간단한 처리
    if not any(word in current_utterance for word in ['나는', '내가', '저는', '제가']):
        # 이전 대화에서 화자가 언급한 경우 복원
        for prev in reversed(previous_utterances[-3:]):
            if any(word in prev for word in ['너는', '네가', '당신은']):
                # 주어 추가
                return '나는 ' + current_utterance
    
    return current_utterance

def load_and_preprocess_data(data_path, emotion_data_path=None):
    """데이터 로드 및 전처리 함수"""
    logger.info(f"데이터 로드 중: {data_path}")
    
    # 대화 데이터 로드 (JSON 형식)
    if data_path.endswith('.json'):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"JSON 데이터 로드 완료: {len(data)}개 항목")
            
            # 대화 및 라벨 추출
            conversations = []
            labels = []
            emotion_tags = []
            
            for item in data:
                if 'texts' in item and 'gaslighting' in item:
                    conversations.append(item['texts'])
                    labels.append(item['gaslighting'])
                    
                    # 감정 태그 추출 (있는 경우)
                    if 'emotions' in item:
                        emotion_tags.append(item['emotions'])
            
            logger.info(f"추출된 대화 수: {len(conversations)}")
            logger.info(f"가스라이팅 라벨 분포: {pd.Series(labels).value_counts().to_dict()}")
            
            return conversations, labels, emotion_tags if emotion_tags else None
        
        except Exception as e:
            logger.error(f"JSON 데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    # CSV 형식 데이터 로드 (기존 코드)
    else:
        # 대화 데이터 로드
        df = pd.read_csv(data_path)
        
        # 감정 데이터 로드 (있는 경우)
        emotion_df = None
        if emotion_data_path and os.path.exists(emotion_data_path):
            logger.info(f"감정 데이터 로드 중: {emotion_data_path}")
            emotion_df = pd.read_csv(emotion_data_path)
        
        # 데이터 기본 정보 로깅
        logger.info(f"원본 데이터 크기: {len(df)}")
        
        # NaN 값 처리
        if 'gaslighting' in df.columns:
            logger.info(f"가스라이팅 라벨 NaN 값 개수: {df['gaslighting'].isna().sum()}")
        
        # NaN 값과 빈 문자열 제거
        df = df.dropna(subset=['conversation_id', 'text'])
        df = df[df['text'].str.strip().str.len() > 0]
        logger.info(f"NaN 및 빈 문자열 제거 후 데이터 크기: {len(df)}")
        
        # 대화 그룹화
        conversations = []
        labels = []
        emotion_tags = []
        
        # 대화 ID별로 그룹화
        for conv_id, group in df.groupby('conversation_id'):
            # 대화 턴 정렬
            group = group.sort_values('turn_id')
            
            # 텍스트 전처리
            texts = []
            previous_texts = []
            
            for _, row in group.iterrows():
                text = preprocess_text(row['text'])
                
                # 생략된 주어/목적어 복원
                text = restore_ellipsis(text, previous_texts, None)
                
                texts.append(text)
                previous_texts.append(text)
            
            # 가스라이팅 라벨 (있는 경우)
            if 'gaslighting' in group.columns:
                # 대화에 가스라이팅이 하나라도 있으면 1, 아니면 0
                label = 1 if group['gaslighting'].any() else 0
            else:
                # 라벨이 없는 경우 (테스트 데이터)
                label = 0
            
            # 감정 태그 (있는 경우)
            if emotion_df is not None:
                conv_emotions = emotion_df[emotion_df['conversation_id'] == conv_id]
                if not conv_emotions.empty:
                    # 턴별 감정 태그 추출
                    turn_emotions = []
                    for turn_id in group['turn_id']:
                        emotion = conv_emotions[conv_emotions['turn_id'] == turn_id]['emotion'].values
                        turn_emotions.append(emotion[0] if len(emotion) > 0 else None)
                    emotion_tags.append(turn_emotions)
                else:
                    emotion_tags.append([None] * len(texts))
            
            conversations.append(texts)
            labels.append(label)
        
        logger.info(f"전처리 후 대화 수: {len(conversations)}")
        logger.info(f"가스라이팅 라벨 분포: {pd.Series(labels).value_counts().to_dict()}")
        
        return conversations, labels, emotion_tags if emotion_df is not None else None

def evaluate_model(model, dataloader, device, simplified=False):
    """모델 평가 함수"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_risk_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 데이터 디바이스 이동
            labels = batch['label'].to(device)
            
            # 모델 예측
            if simplified:
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['emotion_ids'].to(device) if 'emotion_ids' in batch else None
                )
                
                # 간소화된 모델은 'logits'를 반환
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                risk_scores = torch.ones_like(labels).float() * 0.5  # 더미 위험 점수
            else:
                outputs = model(
                    batch['conversation_input_ids'].to(device),
                    batch['conversation_attention_mask'].to(device),
                    batch['emotion_ids'].to(device) if 'emotion_ids' in batch else None
                )
                
                # 전체 모델은 'gaslighting_logits'와 'risk_score'를 반환
                logits = outputs['gaslighting_logits']
                risk_scores = outputs['risk_score']
            
            # 예측 결과 수집
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_risk_scores.extend(risk_scores.cpu().numpy())
    
    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    
    # 위험 점수 평균
    avg_risk_score = np.mean(all_risk_scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'avg_risk_score': avg_risk_score,
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'risk_scores': np.array(all_risk_scores)
    }

def train_gaslighting_detector(args):
    """가스라이팅 탐지 모델 훈련 함수"""
    logger.info("가스라이팅 탐지 모델 훈련 시작")
    
    # 데이터 로드 및 전처리
    conversations, labels, emotion_tags = load_and_preprocess_data(
        args.data_path, 
        args.emotion_data_path
    )
    
    # 데이터 분할
    train_conversations, test_conversations, train_labels, test_labels = train_test_split(
        conversations, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_conversations, val_conversations, train_labels, val_labels = train_test_split(
        train_conversations, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    
    # 감정 태그 분할 (있는 경우)
    train_emotions, val_emotions, test_emotions = None, None, None
    if emotion_tags:
        # 인덱스 기반 분할
        train_idx, test_idx = train_test_split(
            range(len(conversations)), test_size=0.2, stratify=labels, random_state=42
        )
        
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, stratify=[labels[i] for i in train_idx], random_state=42
        )
        
        # 인덱스를 사용하여 감정 태그 분할
        if len(emotion_tags) == len(conversations):
            train_emotions = [emotion_tags[i] for i in train_idx]
            val_emotions = [emotion_tags[i] for i in val_idx]
            test_emotions = [emotion_tags[i] for i in test_idx]
        else:
            logger.warning(f"감정 태그 수({len(emotion_tags)})가 대화 수({len(conversations)})와 일치하지 않습니다. 감정 태그를 사용하지 않습니다.")
    
    # 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    
    train_dataset = GaslightingDataset(
        train_conversations, train_labels, tokenizer, 
        max_seq_length=args.max_seq_length, 
        max_turns=args.max_turns,
        emotion_tags=train_emotions
    )
    
    val_dataset = GaslightingDataset(
        val_conversations, val_labels, tokenizer, 
        max_seq_length=args.max_seq_length, 
        max_turns=args.max_turns,
        emotion_tags=val_emotions
    )
    
    test_dataset = GaslightingDataset(
        test_conversations, test_labels, tokenizer, 
        max_seq_length=args.max_seq_length, 
        max_turns=args.max_turns,
        emotion_tags=test_emotions
    )
    
    logger.info(f"훈련 데이터 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터 크기: {len(val_dataset)}")
    logger.info(f"테스트 데이터 크기: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 모델 초기화
    if args.simplified:
        # 간소화된 BERT-LSTM 모델
        model = BertLstmHybridModel(
            bert_model_name=args.bert_model_name,
            hidden_size=768,
            lstm_hidden_size=256,
            num_classes=2,
            dropout_rate=0.2,
            num_emotions=7,
            emotion_embedding_dim=64,
            num_attention_heads=8
        ).to(args.device)
    else:
        # 전체 가스라이팅 탐지 시스템
        model = GaslightingDetector(
            bert_model_name=args.bert_model_name,
            hidden_size=768,
            lstm_hidden_size=256,
            num_classes=2,
            dropout_rate=0.2,
            num_emotions=7,
            emotion_embedding_dim=64,
            num_attention_heads=8,
            max_conversation_turns=args.max_turns,
            max_seq_length=args.max_seq_length
        ).to(args.device)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(args.device))
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    # 학습
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # 훈련 루프
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            # 데이터 디바이스 이동
            conversation_input_ids = batch['conversation_input_ids'].to(args.device)
            conversation_attention_mask = batch['conversation_attention_mask'].to(args.device)
            emotion_ids = batch['emotion_ids'].to(args.device) if 'emotion_ids' in batch else None
            labels = batch['label'].to(args.device)
            
            # 모델 예측
            if args.simplified:
                outputs = model(
                    batch['input_ids'].to(args.device),
                    batch['attention_mask'].to(args.device),
                    batch['emotion_ids'].to(args.device) if 'emotion_ids' in batch else None
                )
            else:
                outputs = model(
                    batch['conversation_input_ids'].to(args.device),
                    batch['conversation_attention_mask'].to(args.device),
                    batch['emotion_ids'].to(args.device) if 'emotion_ids' in batch else None
                )
            
            # 손실 계산
            if args.simplified:
                loss = criterion(outputs['logits'], labels)
            else:
                loss = criterion(outputs['gaslighting_logits'], labels)
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 검증
        val_metrics = evaluate_model(model, val_loader, args.device, args.simplified)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val Risk Score: {val_metrics['avg_risk_score']:.4f}")
        
        # 최고 성능 모델 저장
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            logger.info(f"새로운 최고 F1 스코어: {best_val_f1:.4f}")
        
        # 조기 종료
        if early_stopping(avg_train_loss, model):
            logger.info("Early stopping triggered")
            break
    
    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 최종 테스트
    test_metrics = evaluate_model(model, test_loader, args.device, args.simplified)
    
    logger.info(f"최종 테스트 결과:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Avg Risk Score: {test_metrics['avg_risk_score']:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), args.output_path)
    logger.info(f"가스라이팅 탐지 모델이 {args.output_path}에 저장되었습니다.")
    
    # 테스트 결과 저장
    test_results = {
        'metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'auc': float(test_metrics['auc']),
            'avg_risk_score': float(test_metrics['avg_risk_score'])
        },
        'predictions': test_metrics['predictions'].tolist(),
        'probabilities': test_metrics['probabilities'].tolist(),
        'risk_scores': test_metrics['risk_scores'].tolist()
    }
    
    with open(args.output_path + '_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"테스트 결과가 {args.output_path}_test_results.json에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='가스라이팅 탐지 모델 훈련')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='대화 데이터 파일 경로')
    parser.add_argument('--emotion_data_path', type=str, default=None,
                        help='감정 태그 데이터 파일 경로 (옵션)')
    parser.add_argument('--output_path', type=str, default='models/gaslighting_detector',
                        help='모델 저장 경로')
    parser.add_argument('--bert_model_name', type=str, default='klue/bert-base',
                        help='BERT 모델 이름')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='학습에 사용할 디바이스')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에폭 수')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='학습률')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='최대 시퀀스 길이')
    parser.add_argument('--max_turns', type=int, default=10,
                        help='최대 대화 턴 수')
    parser.add_argument('--simplified', action='store_true',
                        help='간소화된 모델 사용 여부')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 모델 훈련
    train_gaslighting_detector(args)

if __name__ == '__main__':
    main() 