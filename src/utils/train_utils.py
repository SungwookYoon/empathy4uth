import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import copy
import random
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.nn.utils import clip_grad_norm_

class EarlyStopping:
    """조기 종료 클래스"""
    def __init__(self, patience=3, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            
    def get_best_model(self):
        return self.best_model

class TextDataset(Dataset):
    """텍스트 데이터셋 클래스"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # NaN 값과 빈 문자열 필터링
        valid_indices = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if (isinstance(text, str) and text.strip() and 
                isinstance(label, (int, float, np.integer, np.floating)) and 
                not np.isnan(label)):
                valid_indices.append(i)
        
        # 유효한 데이터만 유지
        self.texts = [texts[i] for i in valid_indices]
        self.labels = [int(labels[i]) for i in valid_indices]  # int 타입으로 변환
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DialogDataset(Dataset):
    """대화 데이터셋 클래스"""
    def __init__(self, queries, responses, empathy_types=None, tokenizer=None, max_length=128, augment=False, augment_prob=0.7):
        self.queries = queries
        self.responses = responses
        self.empathy_types = empathy_types
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("klue/bert-base")
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.queries)
    
    def augment_text(self, text):
        """강화된 텍스트 증강 메서드 - TextDataset과 동일한 메커니즘 사용"""
        augmentation_type = random.randint(0, 5)
        
        # 원본 보존 확률
        if random.random() > self.augment_prob:
            return text
            
        # TextDataset과 동일한 증강 로직 사용
        if augmentation_type == 0:
            # 단어 삭제
            words = text.split()
            if len(words) <= 3:
                return text
                
            delete_count = max(1, int(len(words) * 0.2))
            indices_to_delete = random.sample(range(len(words)), delete_count)
            new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
            return ' '.join(new_words)
            
        elif augmentation_type == 1:
            # 단어 순서 변경
            words = text.split()
            if len(words) <= 3:
                return text
                
            swap_count = max(1, int(len(words) * 0.1))
            for _ in range(swap_count):
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            return ' '.join(words)
            
        elif augmentation_type == 2:
            # 유의어로 대체 (간단한 시뮬레이션)
            common_korean_words = {
                "좋은": ["멋진", "훌륭한", "괜찮은"],
                "나쁜": ["안좋은", "형편없는", "별로인"],
                "학교": ["교실", "학원", "공부하는 곳"],
                "친구": ["동료", "짝꿍", "친구들"],
                "가족": ["부모님", "집안", "가정"],
                "생각": ["의견", "고민", "걱정"],
                "기분": ["감정", "느낌", "상태"],
                "화가": ["분노", "짜증", "불만"],
                "슬픔": ["우울", "서러움", "눈물"],
                "행복": ["기쁨", "즐거움", "신남"]
            }
            
            for word, replacements in common_korean_words.items():
                if word in text and random.random() < 0.5:
                    text = text.replace(word, random.choice(replacements), 1)
            return text
            
        elif augmentation_type == 3:
            # 문장 일부 반복
            words = text.split()
            if len(words) < 2:
                return text
                
            start_idx = random.randint(0, len(words) - 2)
            repeat_length = random.randint(1, min(3, len(words) - start_idx))
            segment_to_repeat = words[start_idx:start_idx + repeat_length]
            
            if random.random() < 0.5:
                words = words[:start_idx] + segment_to_repeat + segment_to_repeat + words[start_idx + repeat_length:]
            else:
                words = words + segment_to_repeat
                
            return ' '.join(words)
            
        elif augmentation_type == 4:
            # 맞춤법/문법 오류 시뮬레이션
            typos = {
                "을": "를", "를": "을",
                "이": "가", "가": "이",
                "은": "는", "는": "은",
                "습니다": "슴니다",
                "했어요": "햇어요",
                "같아요": "갓아요",
                "그리고": "그리구",
                "때문에": "데문에",
                "있어요": "잇어요"
            }
            
            for correct, typo in typos.items():
                if correct in text and random.random() < 0.3:
                    text = text.replace(correct, typo, 1)
            return text
            
        elif augmentation_type == 5:
            # 문장 접두/접미 추가
            prefixes = ["사실은 ", "솔직히 ", "제 생각에는 ", "오늘 ", "요즘 ", "저는 ", ""]
            suffixes = [" 그렇게 생각해요", " 그런 것 같아요", " 그게 걱정이에요", " 어떻게 생각해요?", " 도와주세요", ""]
            
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            
            if prefix == "" and suffix == "":
                if random.random() < 0.5:
                    prefix = random.choice([p for p in prefixes if p != ""])
                else:
                    suffix = random.choice([s for s in suffixes if s != ""])
                    
            return prefix + text + suffix
            
        return text

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        
        # 데이터 증강 적용 (훈련 시에만)
        if self.augment and random.random() < 0.3:  # 30% 확률로 증강
            query = self.augment_text(query)
        
        query_encodings = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        response_encodings = self.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": query_encodings["input_ids"].squeeze(),
            "attention_mask": query_encodings["attention_mask"].squeeze(),
            "token_type_ids": query_encodings["token_type_ids"].squeeze() if "token_type_ids" in query_encodings else None,
            "decoder_input_ids": response_encodings["input_ids"].squeeze(),
            "decoder_attention_mask": response_encodings["attention_mask"].squeeze(),
        }
        
        if self.empathy_types is not None:
            item["empathy_type"] = torch.tensor(self.empathy_types[idx])
            
        return item

def train_classifier(model, train_loader, val_loader=None, epochs=10, lr=5e-5, device="cuda", weight_decay=0.01, class_weights=None):
    """
    분류기 모델 훈련 함수 - 가중치 감소 및 클래스 가중치 지원 추가
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # 클래스 가중치 적용
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # AdamW 옵티마이저와 가중치 감소
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습률 스케줄러 - 워밍업 후 선형 감소
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% 워밍업
        num_training_steps=total_steps
    )
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            if token_type_ids is not None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
                
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 그래디언트 클리핑 (폭발 방지)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_cpu)
            
        train_loss = train_loss / len(train_loader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(all_labels, all_preds))
        
        if val_loader:
            val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping 체크
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                # 가장 좋은 모델 상태를 반환
                return early_stopping.get_best_model(), criterion
    
    # 최종 모델 상태 반환
    return model.state_dict(), criterion

def evaluate_classifier(model, data_loader, criterion=None, device="cuda"):
    """분류기 모델 평가 함수"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if batch.get("token_type_ids") is not None else None
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(all_labels, all_preds))
    
    return avg_loss, accuracy

def train_aspect_sentiment(model, train_loader, val_loader=None, epochs=10, lr=5e-5,
                           device="cuda", weight_decay=0.01,
                           aspect_weights=None, sentiment_weights=None):
    """
    속성 감정 분석 모델 훈련 함수 - 클래스 가중치 지원 추가
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # 속성과 감정 가중치 적용
    if aspect_weights is not None:
        aspect_criterion = torch.nn.CrossEntropyLoss(weight=aspect_weights.to(device))
    else:
        aspect_criterion = torch.nn.CrossEntropyLoss()
        
    if sentiment_weights is not None:
        sentiment_criterion = torch.nn.CrossEntropyLoss(weight=sentiment_weights.to(device))
    else:
        sentiment_criterion = torch.nn.CrossEntropyLoss()
    
    # AdamW 옵티마이저와 가중치 감소
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_aspect_preds = []
        all_aspect_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            aspect_labels = batch["aspect_labels"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            
            optimizer.zero_grad()
            
            if token_type_ids is not None:
                aspect_logits, sentiment_logits = model(input_ids, attention_mask, token_type_ids)
            else:
                aspect_logits, sentiment_logits = model(input_ids, attention_mask)
                
            aspect_loss = aspect_criterion(aspect_logits, aspect_labels)
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            loss = aspect_loss + sentiment_loss
            
            loss.backward()
            
            # 그래디언트 클리핑 (폭발 방지)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            aspect_preds = torch.argmax(aspect_logits, dim=1).detach().cpu().numpy()
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).detach().cpu().numpy()
            
            all_aspect_preds.extend(aspect_preds)
            all_aspect_labels.extend(aspect_labels.cpu().numpy())
            all_sentiment_preds.extend(sentiment_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            
        train_loss = train_loss / len(train_loader)
        aspect_accuracy = accuracy_score(all_aspect_labels, all_aspect_preds)
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Aspect Acc: {aspect_accuracy:.4f}, Sentiment Acc: {sentiment_accuracy:.4f}")
        
        if val_loader:
            val_loss, val_aspect_acc, val_sentiment_acc = evaluate_aspect_sentiment(
                model, val_loader, (aspect_criterion, sentiment_criterion), device
            )
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, "
                  f"Val Aspect Acc: {val_aspect_acc:.4f}, Val Sentiment Acc: {val_sentiment_acc:.4f}")
            
            # Early stopping 체크 - 전체 손실에 기반
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                # 가장 좋은 모델 상태를 반환
                return early_stopping.get_best_model(), (aspect_criterion, sentiment_criterion)
    
    # 최종 모델 상태 반환
    return model.state_dict(), (aspect_criterion, sentiment_criterion)

def evaluate_aspect_sentiment(model, data_loader, criterion=None, device="cuda"):
    """속성 감정 분석 모델 평가 함수"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if batch.get("token_type_ids") is not None else None
            aspect_labels = batch["aspect_labels"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            
            aspect_logits, sentiment_logits = model(input_ids, attention_mask, token_type_ids)
            
            aspect_loss = criterion(aspect_logits, aspect_labels)
            sentiment_loss = criterion(sentiment_logits, sentiment_labels)
            loss = aspect_loss + sentiment_loss
            
            total_loss += loss.item()
            
            aspect_preds = torch.argmax(aspect_logits, dim=1).cpu().numpy()
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            
            all_aspect_preds.extend(aspect_preds)
            all_aspect_labels.extend(aspect_labels.cpu().numpy())
            all_sentiment_preds.extend(sentiment_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    aspect_accuracy = accuracy_score(all_aspect_labels, all_aspect_preds)
    sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    aspect_f1 = f1_score(all_aspect_labels, all_aspect_preds, average='weighted')
    sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
    
    print(f"Aspect F1: {aspect_f1:.4f}, Sentiment F1: {sentiment_f1:.4f}")
    
    return avg_loss, aspect_accuracy, sentiment_accuracy

# 클래스 가중치를 계산하는 함수 추가
def compute_class_weights(labels):
    """클래스 별 가중치 계산 (희소 클래스에 더 높은 가중치 부여)"""
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # 클래스 가중치 계산 (희소 클래스에 더 높은 가중치)
    weights = n_samples / (n_classes * class_counts)
    return torch.FloatTensor(weights)

# 가중 샘플링을 위한 샘플러 생성 함수 추가
def create_weighted_sampler(labels):
    """클래스 불균형을 처리하기 위한 가중 샘플러 생성"""
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    
    # 각 샘플의 가중치 계산
    sample_weights = class_weights[labels]
    
    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler 