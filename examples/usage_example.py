import os
import sys
import json
import argparse

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.integrated_model import EmPathIntegratedModel

def main():
    """EmPath 모델 사용 예제"""
    parser = argparse.ArgumentParser(description="EmPath 모델 사용 예제")
    
    parser.add_argument("--model_dir", type=str, default="models/base_models", help="기본 모델 디렉토리")
    parser.add_argument("--weights_path", type=str, default=None, help="최적 가중치 파일 경로 (optional)")
    parser.add_argument("--model_name", type=str, default="klue/bert-base", help="사전 학습 모델 이름")
    parser.add_argument("--device", type=str, default="cuda", help="모델 실행 장치 (cuda 또는 cpu)")
    parser.add_argument("--input_text", type=str, default=None, help="분석할 텍스트 (없으면 대화형 모드 실행)")
    parser.add_argument("--output_file", type=str, default=None, help="결과 저장 파일 경로 (optional)")
    
    args = parser.parse_args()
    
    # 통합 모델 인스턴스 생성
    print("EmPath 모델 로드 중...")
    model = EmPathIntegratedModel(
        model_dir=args.model_dir,
        model_name=args.model_name,
        device=args.device
    )
    
    # 최적 가중치 로드 (있는 경우)
    if args.weights_path and os.path.exists(args.weights_path):
        print(f"최적 가중치 로드 중: {args.weights_path}")
        with open(args.weights_path, "r") as f:
            weights = json.load(f)
            
        model.crisis_weights = weights["crisis_weights"]
        model.aspect_weights = weights["aspect_weights"]
        model.sentiment_weights = weights["sentiment_weights"]
        model.sentence_type_weights = weights["sentence_type_weights"]
    
    # 명령줄 인자로 텍스트가 제공된 경우
    if args.input_text:
        analyze_and_print(model, args.input_text, args.output_file)
    else:
        # 대화형 모드 실행
        interactive_mode(model, args.output_file)

def analyze_and_print(model, text, output_file=None):
    """텍스트 분석 및 결과 출력"""
    print("\n분석 중...\n")
    
    try:
        # 텍스트 분석
        analysis = model.analyze(text)
        
        # 결과 출력
        print_analysis_results(text, analysis)
        
        # 결과 저장 (파일 경로가 제공된 경우)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "text": text,
                    "analysis": analysis
                }, f, indent=4, ensure_ascii=False)
            print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def interactive_mode(model, output_file=None):
    """대화형 모드"""
    print("\n=== EmPath 대화형 모드 ===")
    print("텍스트를 입력하면 분석 결과를 보여줍니다.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.\n")
    
    results = []
    
    while True:
        text = input("\n텍스트 입력 (종료: exit 또는 quit): ")
        
        if text.lower() in ["exit", "quit"]:
            break
        
        if not text.strip():
            continue
        
        try:
            # 텍스트 분석
            analysis = model.analyze(text)
            
            # 결과 출력
            print_analysis_results(text, analysis)
            
            # 결과 저장을 위해 기록
            results.append({
                "text": text,
                "analysis": analysis
            })
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
    
    # 모든 결과 저장 (파일 경로가 제공된 경우)
    if output_file and results:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n모든 분석 결과가 {output_file}에 저장되었습니다.")

def print_analysis_results(text, analysis):
    """분석 결과 출력"""
    print("\n" + "="*50)
    print("분석 텍스트:")
    print(text)
    print("="*50)
    
    # 1. 위기 분석 결과
    crisis_analysis = analysis["crisis_analysis"]
    print("\n[위기 수준 분석]")
    print(f"최종 위기 수준: {crisis_analysis['final_crisis_level']}")
    print(f"위기 점수: {crisis_analysis['crisis_score']:.4f}")
    
    # 2. 문장 유형 분석 결과
    type_analysis = analysis["type_analysis"]
    print("\n[문장 유형 분석]")
    for type_name, count in type_analysis["type_counts"].items():
        if count > 0:
            print(f"{type_name}: {count}개")
    
    # 3. 속성 감정 분석 결과
    sentiment_analysis = analysis["aspect_sentiment_analysis"]
    print("\n[속성별 감정 분석]")
    for aspect, sentiments in sentiment_analysis["aspect_sentiment_matrix"].items():
        if sum(sentiments.values()) > 0:
            print(f"- {aspect}:")
            for sentiment, count in sentiments.items():
                if count > 0:
                    print(f"  {sentiment}: {count}개")
    
    # 4. 통합 위험 점수
    risk_score = analysis["risk_score"]
    print("\n[통합 위험 점수]")
    print(f"{risk_score:.4f}/1.0")
    
    # 위험 수준 텍스트로 표현
    if risk_score < 0.3:
        risk_level = "낮음"
    elif risk_score < 0.6:
        risk_level = "중간"
    else:
        risk_level = "높음"
    print(f"위험 수준: {risk_level}")
    
    # 5. 공감 응답
    empathy_response = analysis["empathy_response"]
    print("\n[공감 응답]")
    print(f"공감 유형: {empathy_response['empathy_type']}")
    print(f"응답: {empathy_response['response']}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 