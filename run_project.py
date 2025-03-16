import os
import sys
import subprocess
import argparse
import logging
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data/raw",
        "data/samples",
        "data/processed",
        "models/base_models",
        "models/integrated_model",
        "logs",
        "examples",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"디렉토리 생성: {directory}")

def run_command(command, description):
    """명령어 실행"""
    logger.info(f"{description} 실행 중...")
    logger.info(f"명령어: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"{description} 실행 중 오류 발생:")
            logger.error(stderr)
            return False
        
        logger.info(f"{description} 실행 완료")
        return True
    
    except Exception as e:
        logger.error(f"{description} 실행 중 예외 발생: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="EmPath 프로젝트 실행 스크립트")
    
    parser.add_argument("--skip_sampling", action="store_true", help="샘플 추출 단계 건너뛰기")
    parser.add_argument("--skip_base_training", action="store_true", help="기본 모델 훈련 단계 건너뛰기")
    parser.add_argument("--skip_integration", action="store_true", help="통합 모델 평가 단계 건너뛰기")
    parser.add_argument("--skip_testing", action="store_true", help="테스트 단계 건너뛰기")
    parser.add_argument("--device", type=str, default="cuda", help="모델 실행 장치 (cuda 또는 cpu)")
    parser.add_argument("--model_name", type=str, default="klue/bert-base", help="사전 학습 모델 이름")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=3, help="에폭 수")
    
    args = parser.parse_args()
    
    # 시작 시간 기록
    start_time = datetime.now()
    logger.info(f"EmPath 프로젝트 실행 시작: {start_time}")
    
    # 디렉토리 설정
    setup_directories()
    
    # 1. 샘플 추출
    if not args.skip_sampling:
        sample_command = "python scripts/sample_extraction.py"
        if not run_command(sample_command, "샘플 추출"):
            logger.error("샘플 추출 실패, 프로세스 중단")
            return
    else:
        logger.info("샘플 추출 단계 건너뛰기")
    
    # 2. 기본 모델 훈련
    if not args.skip_base_training:
        train_base_command = (
            f"python scripts/train_base_models.py "
            f"--model_name {args.model_name} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--device {args.device}"
        )
        if not run_command(train_base_command, "기본 모델 훈련"):
            logger.error("기본 모델 훈련 실패, 프로세스 중단")
            return
    else:
        logger.info("기본 모델 훈련 단계 건너뛰기")
    
    # 3. 통합 모델 평가 및 최적화
    if not args.skip_integration:
        integrated_command = (
            f"python scripts/train_integrated_model.py "
            f"--model_name {args.model_name} "
            f"--device {args.device} "
            f"--evaluate "
            f"--fine_tune "
            f"--test_examples"
        )
        if not run_command(integrated_command, "통합 모델 평가 및 최적화"):
            logger.error("통합 모델 평가 및 최적화 실패, 프로세스 중단")
            return
    else:
        logger.info("통합 모델 평가 및 최적화 단계 건너뛰기")
    
    # 4. 테스트 예제 실행
    if not args.skip_testing:
        test_command = (
            f"python examples/usage_example.py "
            f"--model_name {args.model_name} "
            f"--device {args.device} "
            f"--weights_path models/integrated_model/optimal_weights.json "
            f"--output_file results/test_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json "
            f"--input_text \"학교에서 친구들이 자꾸 나를 놀려요. 집에 가기가 싫어요.\""
        )
        if not run_command(test_command, "테스트 예제 실행"):
            logger.error("테스트 예제 실행 실패")
    else:
        logger.info("테스트 단계 건너뛰기")
    
    # 종료 시간 및 소요 시간 기록
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(f"EmPath 프로젝트 실행 완료: {end_time}")
    logger.info(f"총 소요 시간: {elapsed_time}")
    
    # 결과 요약
    summary = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time_seconds": elapsed_time.total_seconds(),
        "device": args.device,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps": {
            "sampling": not args.skip_sampling,
            "base_training": not args.skip_base_training,
            "integration": not args.skip_integration,
            "testing": not args.skip_testing
        }
    }
    
    # 요약 정보 저장
    summary_path = f"results/run_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"실행 요약 정보가 {summary_path}에 저장되었습니다.")
    
    logger.info("="*50)
    logger.info("EmPath 프로젝트 실행이 완료되었습니다.")
    logger.info("="*50)

if __name__ == "__main__":
    main() 