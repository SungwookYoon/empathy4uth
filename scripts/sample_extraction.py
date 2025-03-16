import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 데이터 디렉토리 설정
DATA_DIR = "data/raw"
OUTPUT_DIR = "data/samples"

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_counseling_samples():
    """아동·청소년 상담 데이터 샘플 추출"""
    print("상담 데이터 샘플 추출 중...")
    
    # 데이터 로드 (실제로는 파일 경로에 맞게 수정 필요)
    file_path = os.path.join(DATA_DIR, "counseling.csv")
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} 파일이 존재하지 않습니다. 예시 데이터를 생성합니다.")
        # 예시 데이터 생성 (실제 데이터 구조를 모방)
        crisis_levels = ["정상군", "관찰필요", "상담필요", "학대의심", "응급"]
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # 실제 분포를 반영한 가중치
        
        # 최소 권장 샘플 수: 1,200개 (클래스 당 약 240개)
        n_samples = 1200
        
        # 예시 데이터 생성
        data = {
            "id": range(n_samples * 2),  # 더 많은 데이터에서 샘플링하기 위해 2배로 생성
            "text": [f"아동·청소년 상담 텍스트 예시 {i}" for i in range(n_samples * 2)],
            "crisis_level": np.random.choice(crisis_levels, size=n_samples * 2, p=weights)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    # 층화 샘플링 (위기 수준별 비율 유지)
    samples, _ = train_test_split(
        df, 
        train_size=1200,  # 위기 단계 분류에 권장되는 최소 샘플 수
        stratify=df["crisis_level"],
        random_state=42
    )
    
    # 저장
    output_path = os.path.join(OUTPUT_DIR, "counseling_samples.csv")
    samples.to_csv(output_path, index=False)
    print(f"{len(samples)}개 상담 데이터 샘플이 {output_path}에 저장되었습니다.")
    
    return samples

def extract_aspect_sentiment_samples():
    """속성 감정 분석 데이터 샘플 추출"""
    print("속성 감정 분석 데이터 샘플 추출 중...")
    
    # 데이터 로드 (실제로는 파일 경로에 맞게 수정 필요)
    file_path = os.path.join(DATA_DIR, "aspect_sentiment.csv")
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} 파일이 존재하지 않습니다. 예시 데이터를 생성합니다.")
        # 예시 데이터 생성
        aspects = ["자아", "가족", "친구", "학교", "미래"]
        sentiments = ["긍정", "부정", "중립"]
        
        # 최소 권장 샘플 수: 1,500개 (15개 조합당 약 100개)
        n_samples = 1500
        
        data = {
            "id": range(n_samples * 2),
            "text": [f"속성 감정 분석 텍스트 예시 {i}" for i in range(n_samples * 2)],
            "aspect": np.random.choice(aspects, size=n_samples * 2),
            "sentiment": np.random.choice(sentiments, size=n_samples * 2)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    # 층화 샘플링 (속성과 감정 조합 유지)
    df["strat"] = df["aspect"] + "_" + df["sentiment"]
    samples, _ = train_test_split(
        df, 
        train_size=1500,  # 속성 감정 분석에 권장되는 최소 샘플 수
        stratify=df["strat"] if len(df) > 1500 else None,  # 데이터가 충분할 때만 층화 샘플링
        random_state=42
    )
    
    # 'strat' 열 제거
    samples = samples.drop("strat", axis=1)
    
    # 저장
    output_path = os.path.join(OUTPUT_DIR, "aspect_sentiment_samples.csv")
    samples.to_csv(output_path, index=False)
    print(f"{len(samples)}개 속성 감정 분석 데이터 샘플이 {output_path}에 저장되었습니다.")
    
    return samples

def extract_empathy_dialog_samples():
    """공감형 대화 데이터 샘플 추출"""
    print("공감형 대화 데이터 샘플 추출 중...")
    
    # 데이터 로드 (실제로는 파일 경로에 맞게 수정 필요)
    file_path = os.path.join(DATA_DIR, "empathy_dialog.csv")
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} 파일이 존재하지 않습니다. 예시 데이터를 생성합니다.")
        # 예시 데이터 생성
        empathy_types = ["조언", "격려", "위로", "동조"]
        
        # 최소 권장 샘플 수: 2,000개
        n_samples = 2000
        
        data = {
            "id": range(n_samples * 2),
            "query": [f"사용자 질문 예시 {i}" for i in range(n_samples * 2)],
            "response": [f"공감형 응답 예시 {i}" for i in range(n_samples * 2)],
            "empathy_type": np.random.choice(empathy_types, size=n_samples * 2)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    # 층화 샘플링 (공감 유형별 비율 유지)
    samples, _ = train_test_split(
        df, 
        train_size=2000,  # 공감형 대화에 권장되는 최소 샘플 수
        stratify=df["empathy_type"] if len(df) > 2000 else None,  # 데이터가 충분할 때만 층화 샘플링
        random_state=42
    )
    
    # 저장
    output_path = os.path.join(OUTPUT_DIR, "empathy_dialog_samples.csv")
    samples.to_csv(output_path, index=False)
    print(f"{len(samples)}개 공감형 대화 데이터 샘플이 {output_path}에 저장되었습니다.")
    
    return samples

def extract_sentence_types_samples():
    """문장 유형 판단 데이터 샘플 추출"""
    print("문장 유형 판단 데이터 샘플 추출 중...")
    
    # 데이터 로드 (실제로는 파일 경로에 맞게 수정 필요)
    file_path = os.path.join(DATA_DIR, "sentence_types.csv")
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} 파일이 존재하지 않습니다. 예시 데이터를 생성합니다.")
        # 예시 데이터 생성
        sentence_types = ["사실진술", "감정표현", "의견제시", "도움요청", "기타"]
        
        # 최소 권장 샘플 수: 800개 (클래스당 약 160개)
        n_samples = 800
        
        data = {
            "id": range(n_samples * 2),
            "text": [f"문장 유형 판단 텍스트 예시 {i}" for i in range(n_samples * 2)],
            "sentence_type": np.random.choice(sentence_types, size=n_samples * 2)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    # 층화 샘플링 (문장 유형별 비율 유지)
    samples, _ = train_test_split(
        df, 
        train_size=800,  # 문장 유형 판단에 권장되는 최소 샘플 수
        stratify=df["sentence_type"] if len(df) > 800 else None,  # 데이터가 충분할 때만 층화 샘플링
        random_state=42
    )
    
    # 저장
    output_path = os.path.join(OUTPUT_DIR, "sentence_types_samples.csv")
    samples.to_csv(output_path, index=False)
    print(f"{len(samples)}개 문장 유형 판단 데이터 샘플이 {output_path}에 저장되었습니다.")
    
    return samples

if __name__ == "__main__":
    print("EmPath 데이터셋 샘플 추출 시작...")
    
    # 각 데이터셋별 샘플 추출
    counseling_samples = extract_counseling_samples()
    aspect_sentiment_samples = extract_aspect_sentiment_samples()
    empathy_dialog_samples = extract_empathy_dialog_samples()
    sentence_types_samples = extract_sentence_types_samples()
    
    print("모든 샘플 추출 완료!")
    
    # 요약 정보 출력
    print("\n===== 샘플 추출 요약 =====")
    print(f"상담 데이터 샘플: {len(counseling_samples)}개")
    print(f"속성 감정 분석 데이터 샘플: {len(aspect_sentiment_samples)}개")
    print(f"공감형 대화 데이터 샘플: {len(empathy_dialog_samples)}개")
    print(f"문장 유형 판단 데이터 샘플: {len(sentence_types_samples)}개") 