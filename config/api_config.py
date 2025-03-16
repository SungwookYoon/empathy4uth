import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = "gpt-4-turbo-preview"  # 또는 필요한 모델 버전
MAX_TOKENS = 4096
TEMPERATURE = 0.7

# API 요청 설정
REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_DELAY = 1

# 에러 처리
class APIKeyError(Exception):
    pass

def validate_api_key():
    if not OPENAI_API_KEY:
        raise APIKeyError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.") 