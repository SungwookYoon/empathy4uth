import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer

# 감정 태그 매핑
EMOTION_MAP = {
    '기쁨': 0, 
    '슬픔': 1, 
    '분노': 2, 
    '공포': 3, 
    '놀람': 4, 
    '혐오': 5, 
    '중립': 6,
    None: 6  # 감정 태그가 없는 경우 중립으로 처리
}

# 가스라이팅 유형 매핑
GASLIGHTING_TYPE_MAP = {
    'reality_distortion': '현실 왜곡',
    'emotional_manipulation': '감정 조작',
    'blame_shifting': '책임 전가',
    'isolation': '고립',
    'gradual_intensity': '점진적 강도'
}

def preprocess_conversation(conversation, tokenizer=None, max_seq_length=128):
    """
    대화 전처리 함수
    
    Args:
        conversation: 대화 텍스트 리스트
        tokenizer: 토크나이저 (옵션)
        max_seq_length: 최대 시퀀스 길이
        
    Returns:
        전처리된 대화 텍스트 리스트
    """
    processed_texts = []
    previous_texts = []
    
    for text in conversation:
        # 기본 텍스트 전처리
        text = preprocess_text(text)
        
        # 생략된 주어/목적어 복원
        text = restore_ellipsis(text, previous_texts, tokenizer)
        
        processed_texts.append(text)
        previous_texts.append(text)
    
    return processed_texts

def preprocess_text(text):
    """
    텍스트 전처리 함수
    
    Args:
        text: 입력 텍스트
        
    Returns:
        전처리된 텍스트
    """
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

def restore_ellipsis(current_utterance, previous_utterances, tokenizer=None):
    """
    생략된 주어/목적어 복원 함수
    
    Args:
        current_utterance: 현재 발화 텍스트
        previous_utterances: 이전 발화 텍스트 리스트
        tokenizer: 토크나이저 (옵션)
        
    Returns:
        복원된 텍스트
    """
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

def enhance_text_for_gaslighting_detection(text, gaslighting_keywords=None):
    """
    가스라이팅 탐지를 위한 텍스트 강화 함수
    
    Args:
        text: 입력 텍스트
        gaslighting_keywords: 가스라이팅 관련 키워드 리스트 (옵션)
        
    Returns:
        강화된 텍스트
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text:
        return ""
    
    # 기본 가스라이팅 키워드
    if gaslighting_keywords is None:
        gaslighting_keywords = [
            '착각', '오해', '기억', '잘못', '아니야', '그런 적 없어',
            '예민', '과장', '심각', '네 탓', '네가', '너 때문에',
            '다른 사람들은', '모두', '항상', '절대', '너만', '혼자'
        ]
    
    # 키워드 강조
    for keyword in gaslighting_keywords:
        if keyword in text:
            # 키워드 반복으로 강조
            text = text.replace(keyword, f"{keyword} {keyword} {keyword}")
    
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text

def analyze_emotion_transitions(emotion_sequence):
    """
    감정 전이 분석 함수
    
    Args:
        emotion_sequence: 감정 태그 시퀀스
        
    Returns:
        감정 전이 분석 결과 딕셔너리
    """
    if not emotion_sequence or len(emotion_sequence) < 2:
        return {
            'transitions': [],
            'has_negative_shift': False,
            'confidence_drop': 0.0
        }
    
    # 감정 그룹화 (긍정/부정/중립)
    emotion_groups = {
        'positive': ['기쁨'],
        'negative': ['슬픔', '분노', '공포', '혐오'],
        'neutral': ['놀람', '중립', None]
    }
    
    # 감정 그룹 매핑
    def get_emotion_group(emotion):
        for group, emotions in emotion_groups.items():
            if emotion in emotions:
                return group
        return 'neutral'
    
    # 감정 전이 분석
    transitions = []
    for i in range(1, len(emotion_sequence)):
        prev_emotion = emotion_sequence[i-1]
        curr_emotion = emotion_sequence[i]
        
        prev_group = get_emotion_group(prev_emotion)
        curr_group = get_emotion_group(curr_emotion)
        
        transitions.append((prev_group, curr_group))
    
    # 부정적 감정 전이 확인
    has_negative_shift = any(prev == 'positive' and curr == 'negative' for prev, curr in transitions)
    
    # 자신감 하락 지표 계산 (부정적 전이 비율)
    negative_transitions = sum(1 for prev, curr in transitions if prev != 'negative' and curr == 'negative')
    confidence_drop = negative_transitions / len(transitions) if transitions else 0.0
    
    return {
        'transitions': transitions,
        'has_negative_shift': has_negative_shift,
        'confidence_drop': confidence_drop
    }

def prepare_conversation_for_model(conversation, emotion_sequence=None, tokenizer=None, max_seq_length=128, max_turns=10):
    """
    모델 입력을 위한 대화 준비 함수
    
    Args:
        conversation: 대화 텍스트 리스트
        emotion_sequence: 감정 태그 시퀀스 (옵션)
        tokenizer: 토크나이저
        max_seq_length: 최대 시퀀스 길이
        max_turns: 최대 대화 턴 수
        
    Returns:
        모델 입력 텐서 딕셔너리
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    # 대화 턴 처리
    input_ids_list = []
    attention_mask_list = []
    emotion_ids_list = []
    
    # 각 턴별 처리
    for i, turn in enumerate(conversation[:max_turns]):
        # 텍스트 인코딩
        encoding = tokenizer(
            turn,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids_list.append(encoding['input_ids'].squeeze())
        attention_mask_list.append(encoding['attention_mask'].squeeze())
        
        # 감정 태그 처리 (있는 경우)
        if emotion_sequence is not None and i < len(emotion_sequence):
            emotion = emotion_sequence[i]
            emotion_id = EMOTION_MAP.get(emotion, 6)  # 매핑에 없는 경우 중립으로 처리
        else:
            emotion_id = 6  # 감정 태그가 없는 경우 중립으로 처리
        
        emotion_ids_list.append(emotion_id)
    
    # 패딩 처리 (대화 턴이 max_turns보다 적은 경우)
    while len(input_ids_list) < max_turns:
        # 빈 턴 추가
        padding_encoding = tokenizer(
            "",
            add_special_tokens=True,
            max_length=max_seq_length,
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
    
    # 배치 차원 추가
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    emotion_ids = emotion_ids.unsqueeze(0)
    
    return {
        'conversation_input_ids': input_ids,
        'conversation_attention_mask': attention_mask,
        'emotion_ids': emotion_ids
    }

def evaluate_gaslighting_detector(model, test_loader, device):
    """
    가스라이팅 탐지 모델 평가 함수
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스
        
    Returns:
        평가 결과 딕셔너리
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_risk_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 데이터 디바이스 이동
            conversation_input_ids = batch['conversation_input_ids'].to(device)
            conversation_attention_mask = batch['conversation_attention_mask'].to(device)
            emotion_ids = batch['emotion_ids'].to(device) if 'emotion_ids' in batch else None
            labels = batch['label'].to(device)
            
            # 모델 예측
            outputs = model(
                conversation_input_ids,
                conversation_attention_mask,
                emotion_ids
            )
            
            # 출력 키가 모델 유형에 따라 다를 수 있음
            if 'gaslighting_logits' in outputs:
                logits = outputs['gaslighting_logits']
            else:
                logits = outputs['logits']
            
            risk_scores = outputs.get('risk_score', torch.zeros(logits.size(0), 1, device=device))
            
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
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds)
    
    # 위험 점수 분석
    risk_scores_array = np.array(all_risk_scores)
    avg_risk_score = np.mean(risk_scores_array)
    risk_by_class = {
        0: np.mean(risk_scores_array[np.array(all_labels) == 0]),
        1: np.mean(risk_scores_array[np.array(all_labels) == 1])
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'avg_risk_score': avg_risk_score,
        'risk_by_class': risk_by_class,
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'risk_scores': np.array(all_risk_scores),
        'true_labels': np.array(all_labels)
    }

def plot_evaluation_results(eval_results, output_path=None):
    """
    평가 결과 시각화 함수
    
    Args:
        eval_results: 평가 결과 딕셔너리
        output_path: 결과 저장 경로 (옵션)
        
    Returns:
        None
    """
    # 1. 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        eval_results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['정상', '가스라이팅'],
        yticklabels=['정상', '가스라이팅']
    )
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('가스라이팅 탐지 혼동 행렬')
    
    if output_path:
        plt.savefig(f"{output_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 주요 지표 시각화
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = [eval_results[metric] for metric in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color='skyblue')
    
    # 값 표시
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha='center'
        )
    
    plt.ylim(0, 1.1)
    plt.title('가스라이팅 탐지 성능 지표')
    
    if output_path:
        plt.savefig(f"{output_path}_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 위험 점수 분포 시각화
    plt.figure(figsize=(10, 6))
    
    # 클래스별 위험 점수 분포
    risk_scores = eval_results['risk_scores']
    true_labels = eval_results['true_labels']
    
    sns.histplot(
        x=risk_scores[true_labels == 0], 
        color='green', 
        alpha=0.5, 
        label='정상',
        kde=True
    )
    
    sns.histplot(
        x=risk_scores[true_labels == 1], 
        color='red', 
        alpha=0.5, 
        label='가스라이팅',
        kde=True
    )
    
    plt.axvline(x=0.5, color='gray', linestyle='--', label='낮은 위험 임계값')
    plt.axvline(x=0.7, color='orange', linestyle='--', label='중간 위험 임계값')
    plt.axvline(x=0.9, color='darkred', linestyle='--', label='높은 위험 임계값')
    
    plt.xlabel('위험 점수')
    plt.ylabel('빈도')
    plt.title('클래스별 위험 점수 분포')
    plt.legend()
    
    if output_path:
        plt.savefig(f"{output_path}_risk_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def detect_gaslighting_pattern(model, conversation, emotion_sequence=None, tokenizer=None, device='cpu'):
    """
    가스라이팅 패턴 탐지 함수
    
    Args:
        model: 가스라이팅 탐지 모델
        conversation: 대화 텍스트 리스트
        emotion_sequence: 감정 태그 시퀀스 (옵션)
        tokenizer: 토크나이저 (옵션)
        device: 디바이스
        
    Returns:
        탐지 결과 딕셔너리
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    
    # 대화 전처리
    processed_conversation = preprocess_conversation(conversation, tokenizer)
    
    # 모델 입력 준비
    inputs = prepare_conversation_for_model(
        processed_conversation, 
        emotion_sequence, 
        tokenizer
    )
    
    # 디바이스 이동
    for key, tensor in inputs.items():
        inputs[key] = tensor.to(device)
    
    # 모델 예측
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 출력 키가 모델 유형에 따라 다를 수 있음
    if 'gaslighting_logits' in outputs:
        logits = outputs['gaslighting_logits']
    else:
        logits = outputs['logits']
    
    risk_score = outputs.get('risk_score', torch.zeros(1, 1, device=device))
    
    # 예측 결과 처리
    probs = torch.softmax(logits, dim=1)
    gaslighting_prob = probs[0, 1].item()
    prediction = torch.argmax(logits, dim=1).item()
    risk_score_value = risk_score[0, 0].item()
    
    # 위험 수준 결정
    if risk_score_value >= 0.9:
        risk_level = 'high_risk'
    elif risk_score_value >= 0.7:
        risk_level = 'medium_risk'
    elif risk_score_value >= 0.5:
        risk_level = 'low_risk'
    else:
        risk_level = 'no_risk'
    
    # 감정 전이 분석 (있는 경우)
    emotion_analysis = None
    if emotion_sequence:
        emotion_analysis = analyze_emotion_transitions(emotion_sequence)
    
    return {
        'is_gaslighting': bool(prediction),
        'gaslighting_probability': gaslighting_prob,
        'risk_score': risk_score_value,
        'risk_level': risk_level,
        'emotion_analysis': emotion_analysis
    } 