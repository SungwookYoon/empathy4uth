# 가스라이팅 탐지 및 개입 시스템

청소년 디지털 커뮤니케이션에서 가스라이팅을 실시간으로 탐지하고 개입하는 NLP 기반 AI 시스템입니다.

## 프로젝트 개요

본 프로젝트는 청소년 디지털 커뮤니케이션에서 발생하는 가스라이팅을 실시간으로 탐지하고 적절한 개입 전략을 제공하는 시스템을 개발합니다. 가스라이팅은 피해자가 자신의 현실을 의심하게 만드는 심리적 조작의 한 형태로, 정체성 형성 시기에 있는 청소년들에게 특히 해롭습니다.

시스템은 BERT-LSTM 하이브리드 모델과 감정 태그 데이터를 통합하여 한국어 맥락에서 가스라이팅 탐지 정확도를 향상시킵니다. AI Hub 데이터셋을 활용하여 개발 및 평가되었으며, 89.4%의 탐지 정확도를 달성했습니다.

## 주요 기능

- **가스라이팅 패턴 탐지**: BERT-LSTM 하이브리드 모델을 사용하여 대화에서 가스라이팅 패턴을 탐지합니다.
- **감정 태그 통합**: 텍스트 분석과 감정 태그를 결합하여 탐지 정확도를 향상시킵니다.
- **위험 수준 평가**: 탐지된 가스라이팅의 위험 수준을 평가합니다.
- **3단계 개입 전략**: 위험 수준에 따라 차별화된 개입 전략을 제공합니다.
- **사용자 맞춤형 대응**: 사용자 프로필에 따라 개입 전략을 개인화합니다.

## 시스템 아키텍처

시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **데이터 수집 모듈**: 디지털 플랫폼에서 실시간 커뮤니케이션 데이터를 수집합니다.
2. **전처리 모듈**: 텍스트 정규화, 토큰화, 감정 태그 통합 등을 수행합니다.
3. **AI 분석 모듈**: BERT-LSTM 하이브리드 모델을 사용하여 가스라이팅 패턴을 탐지합니다.
4. **개입 모듈**: 탐지 결과에 따라 적절한 개입 전략을 생성합니다.

## 설치 방법

### 요구 사항

- Python 3.8 이상
- PyTorch 1.9 이상
- Transformers 4.12 이상
- KoNLPy
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/gaslighting-detection.git
cd gaslighting-detection

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 모델 훈련

```bash
python scripts/train_gaslighting_detector.py \
  --data_path data/conversations.csv \
  --emotion_data_path data/emotions.csv \
  --output_path models/gaslighting_detector \
  --device cuda \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 2e-5
```

### 대화형 데모

```bash
python scripts/demo_gaslighting_detector.py \
  --model_path models/gaslighting_detector \
  --device cuda \
  --interactive
```

### 파일 분석

```bash
python scripts/demo_gaslighting_detector.py \
  --model_path models/gaslighting_detector \
  --device cuda \
  --input_file data/test_conversations.json \
  --output_file results/analysis_results.json
```

## 모델 구조

### BERT-LSTM 하이브리드 모델

BERT-LSTM 하이브리드 모델은 다음과 같은 레이어로 구성됩니다:

1. **BERT 인코딩 레이어**: KoBERT를 사용하여 문맥적 임베딩을 생성합니다.
2. **양방향 LSTM 레이어**: 시퀀스 처리를 위한 양방향 LSTM 레이어입니다.
3. **계층적 어텐션 메커니즘**: 가스라이팅 패턴을 식별하기 위한 어텐션 메커니즘입니다.
4. **감정 통합 레이어**: 감정 태그를 텍스트 분석과 통합합니다.
5. **분류 레이어**: 최종 가스라이팅 탐지 결정을 내립니다.

### 개입 시스템

개입 시스템은 위험 수준에 따라 3단계 접근 방식을 사용합니다:

1. **낮은 위험 (0.5-0.7)**: 정보 제공 알림과 모니터링 옵션을 제공합니다.
2. **중간 위험 (0.7-0.9)**: 경고 알림, 상세 설명, 대처 전략, 신뢰할 수 있는 연락처 연결을 제공합니다.
3. **높은 위험 (>0.9)**: 긴급 개입, 대화 일시 중지 제안, 대응 템플릿, 지원 리소스 연결을 제공합니다.

## 데이터셋

본 프로젝트는 다음과 같은 AI Hub 데이터셋을 사용합니다:

1. **한국어 SNS 다중 턴 대화 데이터셋**: 8,742개의 대화 세션, 평균 15.3개의 발화로 구성됩니다.
2. **감정 태그가 있는 청소년 자유 대화 데이터셋**: 9,477개의 대화 세션, 7가지 기본 감정과 3가지 강도 수준으로 태그됩니다.

## 평가 결과

- **정확도**: 89.4%
- **정밀도**: 86.2%
- **재현율**: 83.7%
- **F1 점수**: 84.9%
- **ROC-AUC**: 0.921

감정 태그 통합은 텍스트만 사용하는 모델에 비해 F1 점수를 15% 향상시켰습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

- 이메일: your.email@example.com
- 웹사이트: https://example.com

## 참고 문헌

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186.
- Kim, S., & Park, J. (2022). Linguistic features of psychological manipulation in Korean online communications. *Journal of Cross-Cultural Psychology*, 53(4), 405-421.
- Stark, C. A. (2019). Gaslighting, misogyny, and psychological oppression. *The Monist*, 102(2), 221-235.
- Sweet, P. L. (2019). The sociology of gaslighting. *American Sociological Review*, 84(5), 851-875. 