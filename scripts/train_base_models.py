#!/usr/bin/env python
import argparse
import os
import logging
import torch
import numpy as np
import pandas as pd
import json
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
from transformers import AdamW
from sklearn.metrics import f1_score
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base_models import (
    CrisisClassifier, 
    SentenceTypeClassifier, 
    AspectSentimentClassifier, 
    EmpathyResponseGenerator
)
from src.utils.train_utils import (
    TextDataset, 
    DialogDataset, 
    train_classifier, 
    evaluate_classifier, 
    train_aspect_sentiment, 
    evaluate_aspect_sentiment,
    compute_class_weights,
    create_weighted_sampler
)

def train_crisis_classifier(args):
    logger.info("위기 단계 분류 모델 훈련 시작")
    
    # 데이터 로드 및 전처리
    df = pd.read_csv('data/samples/counseling_samples.csv')
    
    # NaN 값 처리
    logger.info(f"원본 데이터 크기: {len(df)}")
    logger.info(f"NaN 값 개수: {df['crisis_level'].isna().sum()}")
    
    # NaN 값과 빈 문자열 제거
    df = df.dropna(subset=['crisis_level', 'text'])
    df = df[df['text'].str.strip().str.len() > 0]
    logger.info(f"NaN 및 빈 문자열 제거 후 데이터 크기: {len(df)}")
    
    # 이진 분류로 변환
    crisis_mapping = {
        '정상': 0,
        '관찰필요': 1,
        '상담필요': 1,
        '위험': 1,
        '긴급': 1
    }
    
    df['crisis_level'] = df['crisis_level'].map(crisis_mapping)
    df = df.dropna(subset=['crisis_level'])  # 매핑 후 NaN 제거
    logger.info(f"위기 단계 클래스 분포: {df['crisis_level'].value_counts().to_dict()}")
    
    # 텍스트 전처리 강화
    crisis_keywords = [
        '자살', '죽고', '싶다', '힘들다', '불안', '우울', '폭력', '학대',
        '위험', '긴급', '고통', '무서워', '두려워', '공포', '절망'
    ]
    
    def enhance_text(text):
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        # 특수문자 처리
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        
        # 위험 키워드 강조
        for keyword in crisis_keywords:
            if keyword in text:
                text = text.replace(keyword, f"{keyword} {keyword} {keyword}")  # 3번 반복하여 강조
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 문장 끝 처리
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    df['text'] = df['text'].apply(enhance_text)
    
    # 빈 문자열이 된 행 제거
    df = df[df['text'].str.len() > 0]
    logger.info(f"전처리 후 최종 데이터 크기: {len(df)}")
    
    # 데이터 분할
    texts = df['text'].tolist()
    labels = df['crisis_level'].astype(int).tolist()  # int 타입으로 변환
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    logger.info(f"훈련 데이터 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터 크기: {len(val_dataset)}")
    logger.info(f"테스트 데이터 크기: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 모델 초기화
    model = SimplifiedClassifier(num_classes=2).to(args.device)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(args.device))
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    # 학습
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['input_ids'].to(args.device),
                batch['attention_mask'].to(args.device)
            )
            
            loss = criterion(outputs, batch['labels'].to(args.device))
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 훈련 메트릭 수집
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(batch['labels'].cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        
        # 검증
        val_metrics = evaluate_binary_classifier(model, val_loader, criterion, args.device)
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # 최고 성능 모델 저장
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            logger.info(f"새로운 최고 F1 스코어: {best_val_f1:.4f}")
        
        if early_stopping(val_metrics['loss'], model):
            logger.info("Early stopping triggered")
            break
    
    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 최종 테스트
    test_metrics = evaluate_binary_classifier(model, test_loader, criterion, args.device)
    logger.info(f"최종 테스트 결과:")
    logger.info(f"Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), 'models/crisis_model')
    logger.info("위기 단계 분류 모델이 models/crisis_model에 저장되었습니다.")

def train_sentence_type_classifier(args):
    logger.info("문장 유형 분류 모델 훈련 시작")
    
    # 데이터 로드 및 전처리
    df = pd.read_csv('data/samples/sentence_types_samples.csv')
    
    # NaN 값 처리
    logger.info(f"원본 데이터 크기: {len(df)}")
    logger.info(f"NaN 값 개수: {df['sentence_type'].isna().sum()}")
    
    # NaN 값과 빈 문자열 제거
    df = df.dropna(subset=['sentence_type', 'text'])
    df = df[df['text'].str.strip().str.len() > 0]
    logger.info(f"NaN 및 빈 문자열 제거 후 데이터 크기: {len(df)}")
    
    # 이진 분류로 변환 (감정표현 vs 도움요청)
    df = df[df['sentence_type'].isin(['감정표현', '도움요청'])]
    logger.info(f"이진 분류 필터링 후 데이터 크기: {len(df)}")
    
    sentence_mapping = {
        '감정표현': 0,
        '도움요청': 1
    }
    
    df['sentence_type'] = df['sentence_type'].map(sentence_mapping)
    df = df.dropna(subset=['sentence_type'])  # 매핑 후 NaN 제거
    logger.info(f"문장 유형 클래스 분포: {df['sentence_type'].value_counts().to_dict()}")
    
    # 텍스트 전처리 강화
    emotion_keywords = ['슬프', '화나', '불안', '걱정', '두려', '외롭', '우울', '힘들']
    help_keywords = ['도와', '어떡', '조언', '상담', '해결', '방법']
    
    def enhance_text(text):
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        # 특수문자 처리
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        
        # 키워드 강조
        for keyword in emotion_keywords + help_keywords:
            if keyword in text:
                text = text.replace(keyword, f"{keyword} {keyword}")
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 문장 끝 처리
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    df['text'] = df['text'].apply(enhance_text)
    
    # 빈 문자열이 된 행 제거
    df = df[df['text'].str.len() > 0]
    logger.info(f"전처리 후 최종 데이터 크기: {len(df)}")
    
    # 데이터 분할
    texts = df['text'].tolist()
    labels = df['sentence_type'].astype(int).tolist()  # int 타입으로 변환
    
    # NaN 값이 있는지 최종 확인
    if any(pd.isna(label) for label in labels):
        logger.warning("라벨에 NaN 값이 있습니다. 제거합니다.")
        valid_indices = [i for i, label in enumerate(labels) if not pd.isna(label)]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # 토크나이저 및 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    logger.info(f"훈련 데이터 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터 크기: {len(val_dataset)}")
    logger.info(f"테스트 데이터 크기: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 모델 초기화
    model = SimplifiedClassifier(num_classes=2).to(args.device)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(args.device))
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    # 학습
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['input_ids'].to(args.device),
                batch['attention_mask'].to(args.device)
            )
            
            loss = criterion(outputs, batch['labels'].to(args.device))
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 훈련 메트릭 수집
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(batch['labels'].cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        
        # 검증
        val_metrics = evaluate_binary_classifier(model, val_loader, criterion, args.device)
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # 최고 성능 모델 저장
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            logger.info(f"새로운 최고 F1 스코어: {best_val_f1:.4f}")
        
        if early_stopping(val_metrics['loss'], model):
            logger.info("Early stopping triggered")
            break
    
    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 최종 테스트
    test_metrics = evaluate_binary_classifier(model, test_loader, criterion, args.device)
    logger.info(f"최종 테스트 결과:")
    logger.info(f"Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), 'models/sentence_type_model')
    logger.info("문장 유형 분류 모델이 models/sentence_type_model에 저장되었습니다.")

def evaluate_binary_classifier(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            loss = criterion(outputs, batch['labels'].to(device))
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class SimplifiedClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_aspect_sentiment_model(args):
    """관점 감성 분석 모델 훈련"""
    logger.info("관점 감성 분석 모델 훈련 시작")
    
    # 데이터 로드 - 파일 경로 수정
    data_path = os.path.join('data', 'samples', 'aspect_sentiment_samples.csv')
    df = pd.read_csv(data_path)
    
    # 클래스 분포 로깅
    aspect_counts = df['aspect'].value_counts().to_dict()
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    logger.info(f"관점 클래스 분포: {aspect_counts}")
    logger.info(f"감성 클래스 분포: {sentiment_counts}")
    
    # 라벨 매핑
    aspect_mapping = {
        '자신': 0,
        '가족': 1,
        '친구': 2,
        '학교': 3,
        '사회': 4,
        '기타': 5
    }
    
    sentiment_mapping = {
        '부정': 0,
        '중립': 1,
        '긍정': 2
    }
    
    # 라벨 매핑 저장
    model_dir = os.path.join('models', 'aspect_sentiment_model')
    os.makedirs(model_dir, exist_ok=True)
    
    label_mapping = {
        'aspect': aspect_mapping,
        'sentiment': sentiment_mapping
    }
    
    with open(os.path.join(model_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    
    # 라벨 변환
    df['aspect_label'] = df['aspect'].map(aspect_mapping)
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    
    # 데이터 분할 - 8:1:1 비율
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df[['aspect_label', 'sentiment_label']].apply(lambda x: str(x['aspect_label']) + '_' + str(x['sentiment_label']), axis=1)
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df[['aspect_label', 'sentiment_label']].apply(lambda x: str(x['aspect_label']) + '_' + str(x['sentiment_label']), axis=1)
    )
    
    # 관점 클래스 가중치 계산
    aspect_weights = compute_class_weights(train_df['aspect_label'].values)
    sentiment_weights = compute_class_weights(train_df['sentiment_label'].values)
    
    logger.info(f"관점 클래스 가중치: {aspect_weights}")
    logger.info(f"감성 클래스 가중치: {sentiment_weights}")
    
    # 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    # 데이터셋 클래스 - 관점 및 감성 분석 태스크에 맞게 수정
    class AspectSentimentDataset(TextDataset):
        def __getitem__(self, idx):
            text = self.texts[idx]
            
            if self.augment and np.random.rand() < 0.5:
                text = self.augment_text(text)
            
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }
            
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
            if self.labels is not None:
                aspect_label = self.labels[0][idx]
                sentiment_label = self.labels[1][idx]
                item['aspect_labels'] = torch.tensor(aspect_label, dtype=torch.long)
                item['sentiment_labels'] = torch.tensor(sentiment_label, dtype=torch.long)
            
            return item
    
    # 데이터셋 생성 - 데이터 증강 활성화
    train_dataset = AspectSentimentDataset(
        texts=train_df['text'].tolist(),
        labels=(train_df['aspect_label'].values, train_df['sentiment_label'].values),
        tokenizer=tokenizer,
        max_length=128,
        augment=True  # 데이터 증강 활성화
    )
    
    val_dataset = AspectSentimentDataset(
        texts=val_df['text'].tolist(),
        labels=(val_df['aspect_label'].values, val_df['sentiment_label'].values),
        tokenizer=tokenizer,
        max_length=128
    )
    
    test_dataset = AspectSentimentDataset(
        texts=test_df['text'].tolist(),
        labels=(test_df['aspect_label'].values, test_df['sentiment_label'].values),
        tokenizer=tokenizer,
        max_length=128
    )
    
    # 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 단순 셔플링 (복합 샘플링 대체)
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 모델 초기화
    model = AspectSentimentClassifier(
        aspect_classes=len(aspect_mapping),
        sentiment_classes=len(sentiment_mapping)
    )
    
    # 모델 훈련
    model_state_dict, criterion = train_aspect_sentiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=args.device,
        weight_decay=0.01,  # L2 정규화
        aspect_weights=aspect_weights,
        sentiment_weights=sentiment_weights
    )
    
    # 모델에 가중치 적용
    model.load_state_dict(model_state_dict)
    
    # 테스트 성능 평가
    test_loss, aspect_acc, sentiment_acc = evaluate_aspect_sentiment(
        model, test_loader, criterion, args.device
    )
    
    logger.info(f"최종 테스트 결과 - Loss: {test_loss:.4f}, 관점 정확도: {aspect_acc:.4f}, 감성 정확도: {sentiment_acc:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    logger.info(f"관점 감성 분석 모델이 {model_dir}에 저장되었습니다.")

def train_empathy_model(args):
    """공감 응답 생성 모델 훈련"""
    logger.info("공감 응답 생성 모델 훈련 시작")
    
    # 간소화된 예제에서는 실제 훈련 생략
    if args.simplified:
        logger.info("간소화된 예제에서는 공감 응답 생성 모델 훈련을 생략합니다.")
        
        # 모델 디렉토리 생성
        model_dir = os.path.join('models', 'empathy_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # 빈 모델 저장 (실제 모델이 아님)
        model = EmpathyResponseGenerator(model_name=args.model_name)
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        
        # 공감 유형 매핑
        empathy_mapping = {
            '격려': 0,
            '공감': 1,
            '위로': 2,
            '조언': 3,
            '질문': 4
        }
        
        # 매핑 저장
        with open(os.path.join(model_dir, 'empathy_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(empathy_mapping, f, ensure_ascii=False, indent=4)
        
        logger.info(f"공감 응답 생성 모델이 {model_dir}에 저장되었습니다.")
        return
    
    # 실제 훈련 로직 (간소화된 예제에서는 실행되지 않음)
    # 데이터 로드 - 파일 경로 수정
    data_path = os.path.join('data', 'samples', 'empathy_dialog_samples.csv')
    df = pd.read_csv(data_path)
    
    # 공감 유형 매핑
    empathy_mapping = {
        '격려': 0,
        '공감': 1,
        '위로': 2,
        '조언': 3,
        '질문': 4
    }
    
    # 모델 디렉토리 생성 및 매핑 저장
    model_dir = os.path.join('models', 'empathy_model')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'empathy_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(empathy_mapping, f, ensure_ascii=False, indent=4)
    
    # 공감 유형 ID 변환
    df['empathy_id'] = df['empathy_type'].map(empathy_mapping)
    
    # 데이터 분할 (간소화를 위해 8:2로 분할)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['empathy_type'])
    
    # 토크나이저
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    # 데이터셋 생성
    train_dataset = DialogDataset(
        queries=train_df['query'].tolist(),
        responses=train_df['response'].tolist(),
        empathy_types=train_df['empathy_id'].values,
        tokenizer=tokenizer,
        max_length=128,
        augment=True
    )
    
    val_dataset = DialogDataset(
        queries=val_df['query'].tolist(),
        responses=val_df['response'].tolist(),
        empathy_types=val_df['empathy_id'].values,
        tokenizer=tokenizer,
        max_length=128
    )
    
    # 데이터로더
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 모델 생성
    model = EmpathyResponseGenerator(model_name=args.model_name)
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    logger.info(f"공감 응답 생성 모델이 {model_dir}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='EmPath 기본 모델 훈련')
    parser.add_argument('--model_name', type=str, default="klue/bert-base", help='사전 훈련된 모델 이름')
    parser.add_argument('--epochs', type=int, default=10, help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='학습률')
    parser.add_argument('--device', type=str, default='cuda', help='훈련 장치 (cuda/cpu)')
    parser.add_argument('--simplified', action='store_true', help='간소화된 예제 실행')
    
    args = parser.parse_args()
    
    # 시드 고정
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    
    # 위기 단계 분류 모델 훈련
    train_crisis_classifier(args)
    
    # 문장 유형 분류 모델 훈련
    train_sentence_type_classifier(args)
    
    # 속성 감정 분석 모델 훈련
    train_aspect_sentiment_model(args)
    
    # 공감 응답 생성 모델 훈련 (간소화)
    train_empathy_model(args)
    
    logger.info("모든 기본 모델 훈련이 완료되었습니다.")

if __name__ == "__main__":
    main() 