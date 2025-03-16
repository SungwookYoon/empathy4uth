#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
import random

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gaslighting_dataset_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 가스라이팅 패턴 키워드 및 표현
GASLIGHTING_PATTERNS = [
    # 현실 부정 (Denying Reality)
    r"그런 적 없어", r"그런 말 한 적 없어", r"네가 잘못 들은 거야", r"네가 기억을 잘못하고 있어",
    r"그런 일 없었어", r"네 상상이야", r"네가 예민한 거야", r"네가 오해한 거야",
    
    # 책임 전가 (Shifting Blame)
    r"네가 날 화나게 했잖아", r"네 탓이야", r"네가 그렇게 행동하니까 그런 거지", 
    r"내가 그럴 수밖에 없었어", r"네가 강요했잖아", r"선택의 여지가 없었어",
    
    # 감정 조작 (Emotional Manipulation)
    r"너무 예민하게 굴지 마", r"농담도 못 알아듣니", r"너무 심각하게 받아들이지 마",
    r"네가 너무 감정적이야", r"진정해", r"왜 이렇게 예민해", r"그냥 장난이었어",
    
    # 혼란 유발 (Confusion Tactics)
    r"내가 언제 그랬어\?", r"네가 미쳤나 봐", r"말도 안 되는 소리 하지 마",
    r"네가 착각하고 있어", r"말이 안 통하네", r"이해가 안 돼",
    
    # 비난과 비하 (Criticism and Belittling)
    r"네가 항상 이래", r"너는 늘 실수해", r"너는 왜 이렇게 못하니", 
    r"다른 사람들은 다 할 수 있는데", r"너만 이해 못 하네", r"너는 항상 그래",
    
    # 고립화 (Isolation)
    r"걔는 널 싫어해", r"걔가 너한테 그런 말 했어", r"다른 사람들도 널 이상하게 봐",
    r"너랑 얘기하면 다들 그래", r"너만 그렇게 생각해", r"다른 사람들은 다 이해해",
    
    # 투사 (Projection)
    r"네가 날 의심하는 거야", r"네가 날 통제하려고 해", r"네가 나한테 거짓말하잖아",
    r"네가 날 이용하려는 거지", r"네가 날 조종하려고 해"
]

# 감정 표현 패턴
EMOTION_PATTERNS = {
    "기쁨": [r"행복해", r"좋아", r"신나", r"즐거워", r"기뻐"],
    "슬픔": [r"슬퍼", r"우울해", r"속상해", r"마음이 아파", r"눈물이 나"],
    "분노": [r"화가 나", r"짜증나", r"열받아", r"미쳐버리겠어", r"화가 머리 끝까지 나"],
    "불안": [r"걱정돼", r"불안해", r"긴장돼", r"두려워", r"무서워"],
    "당혹": [r"당황스러워", r"어리둥절해", r"혼란스러워", r"이해가 안 돼", r"어안이 벙벙해"],
    "죄책감": [r"미안해", r"후회돼", r"죄책감이 들어", r"잘못했어", r"양심이 찔려"],
    "무력감": [r"포기하고 싶어", r"아무것도 할 수 없어", r"희망이 없어", r"내 탓인 것 같아", r"어떻게 해야 할지 모르겠어"]
}

def load_empathy_dialog_data(data_path):
    """공감 대화 데이터 로드 함수"""
    logger.info(f"공감 대화 데이터 로드 중: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"로드된 데이터 크기: {len(df)}")
    
    return df

def load_counseling_data(data_path):
    """상담 데이터 로드 함수"""
    logger.info(f"상담 데이터 로드 중: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"로드된 데이터 크기: {len(df)}")
    
    return df

def process_empathy_dialog_samples(df):
    """공감 대화 샘플 데이터 처리 함수"""
    logger.info("공감 대화 샘플 데이터 처리 중...")
    
    conversations = []
    
    # 대화 ID별로 그룹화
    if 'conversation_id' in df.columns and 'text' in df.columns:
        for conv_id, group in df.groupby('conversation_id'):
            # 대화 턴 정렬 (turn_id가 있는 경우)
            if 'turn_id' in group.columns:
                group = group.sort_values('turn_id')
            
            texts = group['text'].tolist()
            
            # 최소 4턴 이상의 대화만 포함
            if len(texts) >= 4:
                # 메타데이터 추출
                metadata = {}
                if 'emotion' in group.columns:
                    metadata['emotion'] = group['emotion'].iloc[0]
                if 'speaker_id' in group.columns:
                    metadata['speaker_id'] = group['speaker_id'].iloc[0]
                
                conversations.append({
                    'conversation_id': conv_id,
                    'texts': texts,
                    **metadata
                })
    
    logger.info(f"처리된 공감 대화 수: {len(conversations)}")
    return conversations

def process_counseling_samples(df):
    """상담 샘플 데이터 처리 함수"""
    logger.info("상담 샘플 데이터 처리 중...")
    
    conversations = []
    
    # 각 상담 텍스트를 대화로 변환
    for _, row in df.iterrows():
        # 텍스트를 문장 단위로 분리하여 대화로 변환
        text = row['text']
        if pd.isna(text) or not isinstance(text, str):
            continue
            
        # 문장 분리 (마침표, 물음표, 느낌표 기준)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 최소 4문장 이상인 경우만 포함
        if len(sentences) >= 4:
            # 메타데이터 추출
            metadata = {}
            if 'crisis_level' in row:
                metadata['crisis_level'] = row['crisis_level']
            if 'id' in row:
                metadata['original_id'] = row['id']
            
            conversations.append({
                'conversation_id': f"counseling_{row.get('id', random.randint(1000, 9999))}",
                'texts': sentences,
                **metadata
            })
    
    logger.info(f"처리된 상담 대화 수: {len(conversations)}")
    return conversations

def load_raw_empathy_dialog_files(directory):
    """원본 공감 대화 JSON 파일 로드 함수"""
    logger.info(f"원본 공감 대화 파일 로드 중: {directory}")
    
    conversations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and filename.startswith('Empathy_'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 대화 추출
                    if 'dialog' in data:
                        conversation = []
                        for turn in data['dialog']:
                            if 'text' in turn:
                                conversation.append(turn['text'])
                        
                        if len(conversation) >= 4:  # 최소 4턴 이상의 대화만 포함
                            conversations.append({
                                'conversation_id': filename.replace('.json', ''),
                                'texts': conversation,
                                'emotion': data.get('info', {}).get('speaker_emotion', ''),
                                'relation': data.get('info', {}).get('relation', '')
                            })
            except Exception as e:
                logger.error(f"파일 로드 중 오류 발생: {file_path}, {str(e)}")
    
    logger.info(f"로드된 대화 수: {len(conversations)}")
    return conversations

def load_raw_counseling_files(directory):
    """원본 상담 JSON 파일 로드 함수"""
    logger.info(f"원본 상담 파일 로드 중: {directory}")
    
    conversations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 상담 대화 추출 (구조에 맞게 수정 필요)
                    if 'list' in data:
                        conversation = []
                        for item in data.get('list', []):
                            if isinstance(item, dict) and '문항' in item:
                                conversation.append(f"질문: {item['문항']}")
                                if '임상가코멘트' in item:
                                    conversation.append(f"답변: {item.get('임상가코멘트', '')}")
                        
                        if len(conversation) >= 4:  # 최소 4턴 이상의 대화만 포함
                            conversations.append({
                                'conversation_id': data.get('info', {}).get('ID', filename.replace('.json', '')),
                                'texts': conversation,
                                'age': data.get('info', {}).get('나이', 0),
                                'gender': data.get('info', {}).get('성별', ''),
                                'crisis_level': data.get('info', {}).get('위기단계', '')
                            })
            except Exception as e:
                logger.error(f"파일 로드 중 오류 발생: {file_path}, {str(e)}")
    
    logger.info(f"로드된 상담 수: {len(conversations)}")
    return conversations

def detect_gaslighting_in_conversation(conversation):
    """대화에서 가스라이팅 패턴 탐지 함수"""
    gaslighting_count = 0
    gaslighting_turns = []
    
    for i, text in enumerate(conversation):
        for pattern in GASLIGHTING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                gaslighting_count += 1
                gaslighting_turns.append(i)
                break
    
    # 가스라이팅 비율 계산 (전체 대화 턴 대비)
    gaslighting_ratio = gaslighting_count / len(conversation) if len(conversation) > 0 else 0
    
    # 가스라이팅 판단 (비율이 0.2 이상이거나 연속된 가스라이팅 패턴이 있는 경우)
    is_gaslighting = gaslighting_ratio >= 0.2
    
    # 연속된 가스라이팅 패턴 확인
    for i in range(len(gaslighting_turns) - 1):
        if gaslighting_turns[i+1] - gaslighting_turns[i] <= 2:  # 2턴 이내에 연속된 가스라이팅
            is_gaslighting = True
            break
    
    return is_gaslighting, gaslighting_ratio, gaslighting_turns

def detect_emotion_in_text(text):
    """텍스트에서 감정 탐지 함수"""
    emotions = {}
    
    for emotion, patterns in EMOTION_PATTERNS.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        if count > 0:
            emotions[emotion] = count
    
    # 가장 많이 탐지된 감정 반환
    if emotions:
        return max(emotions.items(), key=lambda x: x[1])[0]
    else:
        return None

def create_gaslighting_dataset(conversations, output_path, add_synthetic=True):
    """가스라이팅 데이터셋 생성 함수"""
    logger.info("가스라이팅 데이터셋 생성 중...")
    
    dataset = []
    gaslighting_count = 0
    non_gaslighting_count = 0
    
    for conv in tqdm(conversations):
        texts = conv['texts']
        
        # 대화에서 가스라이팅 패턴 탐지
        is_gaslighting, gaslighting_ratio, gaslighting_turns = detect_gaslighting_in_conversation(texts)
        
        # 각 턴별 감정 탐지
        emotions = []
        for text in texts:
            emotion = detect_emotion_in_text(text)
            emotions.append(emotion)
        
        # 데이터셋에 추가
        dataset.append({
            'conversation_id': conv.get('conversation_id', ''),
            'texts': texts,
            'emotions': emotions,
            'gaslighting': 1 if is_gaslighting else 0,
            'gaslighting_ratio': gaslighting_ratio,
            'gaslighting_turns': gaslighting_turns,
            'metadata': {k: v for k, v in conv.items() if k not in ['conversation_id', 'texts']}
        })
        
        if is_gaslighting:
            gaslighting_count += 1
        else:
            non_gaslighting_count += 1
    
    logger.info(f"원본 데이터: 가스라이팅 {gaslighting_count}, 비가스라이팅 {non_gaslighting_count}")
    
    # 합성 데이터 생성 (필요한 경우)
    if add_synthetic and gaslighting_count < non_gaslighting_count * 0.3:  # 가스라이팅 데이터가 30% 미만인 경우
        synthetic_count = min(non_gaslighting_count // 2, 500)  # 최대 500개 또는 비가스라이팅의 절반
        logger.info(f"합성 가스라이팅 데이터 {synthetic_count}개 생성 중...")
        
        synthetic_dataset = create_synthetic_gaslighting_data(conversations, synthetic_count)
        dataset.extend(synthetic_dataset)
        gaslighting_count += len(synthetic_dataset)
    
    logger.info(f"최종 데이터셋: 가스라이팅 {gaslighting_count}, 비가스라이팅 {non_gaslighting_count}, 총 {len(dataset)}개")
    
    # 데이터셋 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"데이터셋 저장 완료: {output_path}")
    
    # 통계 정보 저장
    stats = {
        'total_conversations': len(dataset),
        'gaslighting_count': gaslighting_count,
        'non_gaslighting_count': non_gaslighting_count,
        'gaslighting_ratio': gaslighting_count / len(dataset) if len(dataset) > 0 else 0,
        'synthetic_count': len(synthetic_dataset) if add_synthetic and 'synthetic_dataset' in locals() else 0
    }
    
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return dataset

def create_synthetic_gaslighting_data(conversations, count):
    """합성 가스라이팅 데이터 생성 함수"""
    logger.info(f"합성 가스라이팅 데이터 {count}개 생성 중...")
    
    synthetic_data = []
    non_gaslighting_convs = [conv for conv in conversations if detect_gaslighting_in_conversation(conv['texts'])[0] == False]
    
    if len(non_gaslighting_convs) == 0:
        logger.warning("합성 데이터 생성을 위한 비가스라이팅 대화가 없습니다.")
        return synthetic_data
    
    for i in range(count):
        # 랜덤하게 비가스라이팅 대화 선택
        conv = random.choice(non_gaslighting_convs)
        texts = conv['texts'].copy()
        
        # 가스라이팅 패턴 삽입
        num_insertions = random.randint(2, min(5, len(texts) // 2))
        insertion_positions = sorted(random.sample(range(len(texts)), num_insertions))
        
        gaslighting_turns = []
        for pos in insertion_positions:
            pattern = random.choice(GASLIGHTING_PATTERNS)
            
            # 기존 텍스트에 가스라이팅 패턴 추가
            if random.random() < 0.5:  # 50% 확률로 기존 텍스트 대체
                texts[pos] = re.sub(r'[.!?]', '', pattern) + "."
            else:  # 50% 확률로 기존 텍스트에 추가
                texts[pos] += " " + re.sub(r'[.!?]', '', pattern) + "."
            
            gaslighting_turns.append(pos)
        
        # 각 턴별 감정 탐지
        emotions = []
        for text in texts:
            emotion = detect_emotion_in_text(text)
            emotions.append(emotion)
        
        # 합성 데이터 추가
        synthetic_data.append({
            'conversation_id': f"synthetic_{i}_{conv.get('conversation_id', '')}",
            'texts': texts,
            'emotions': emotions,
            'gaslighting': 1,  # 가스라이팅으로 표시
            'gaslighting_ratio': num_insertions / len(texts),
            'gaslighting_turns': gaslighting_turns,
            'metadata': {
                'synthetic': True,
                'original_id': conv.get('conversation_id', ''),
                **{k: v for k, v in conv.items() if k not in ['conversation_id', 'texts']}
            }
        })
    
    logger.info(f"합성 가스라이팅 데이터 {len(synthetic_data)}개 생성 완료")
    return synthetic_data

def main():
    parser = argparse.ArgumentParser(description='가스라이팅 데이터셋 생성 스크립트')
    parser.add_argument('--empathy_data', type=str, default='data/samples/empathy_dialog_samples.csv',
                        help='공감 대화 데이터 경로')
    parser.add_argument('--counseling_data', type=str, default='data/samples/counseling_samples.csv',
                        help='상담 데이터 경로')
    parser.add_argument('--raw_empathy_dir', type=str, default='data/raw/empathy_dialog',
                        help='원본 공감 대화 디렉토리 경로')
    parser.add_argument('--raw_counseling_dir', type=str, default='data/raw/counseling',
                        help='원본 상담 디렉토리 경로')
    parser.add_argument('--output_path', type=str, default='data/processed/gaslighting/gaslighting_dataset.json',
                        help='출력 데이터셋 경로')
    parser.add_argument('--add_synthetic', action='store_true',
                        help='합성 가스라이팅 데이터 추가 여부')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 데이터 로드
    conversations = []
    
    # 샘플 데이터 로드 및 처리
    if os.path.exists(args.empathy_data):
        empathy_df = load_empathy_dialog_data(args.empathy_data)
        empathy_conversations = process_empathy_dialog_samples(empathy_df)
        conversations.extend(empathy_conversations)
        logger.info(f"공감 대화 샘플 데이터 {len(empathy_conversations)}개 추가됨")
    
    if os.path.exists(args.counseling_data):
        counseling_df = load_counseling_data(args.counseling_data)
        counseling_conversations = process_counseling_samples(counseling_df)
        conversations.extend(counseling_conversations)
        logger.info(f"상담 샘플 데이터 {len(counseling_conversations)}개 추가됨")
    
    # 원본 데이터 로드
    if os.path.exists(args.raw_empathy_dir):
        raw_empathy_conversations = load_raw_empathy_dialog_files(args.raw_empathy_dir)
        conversations.extend(raw_empathy_conversations)
        logger.info(f"원본 공감 대화 데이터 {len(raw_empathy_conversations)}개 추가됨")
    
    if os.path.exists(args.raw_counseling_dir):
        raw_counseling_conversations = load_raw_counseling_files(args.raw_counseling_dir)
        conversations.extend(raw_counseling_conversations)
        logger.info(f"원본 상담 데이터 {len(raw_counseling_conversations)}개 추가됨")
    
    # 데이터셋 생성
    if conversations:
        logger.info(f"총 {len(conversations)}개의 대화 데이터로 가스라이팅 데이터셋 생성 시작")
        create_gaslighting_dataset(conversations, args.output_path, args.add_synthetic)
    else:
        logger.error("로드된 대화 데이터가 없습니다.")

if __name__ == '__main__':
    main() 