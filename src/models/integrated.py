import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout
from transformers import TFBertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Any, Tuple
import json

class EmPathSystem:
    def __init__(self, config_path: str):
        """
        EmPathSystem 초기화

        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 토크나이저 및 기본 모델 로드
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.sentence_type_model = self._load_sentence_type_model()
        self.neologism_model = self._load_neologism_model()
        self.aspect_sentiment_model = self._load_aspect_sentiment_model()
        self.empathy_model = self._load_empathy_model()

        # 통합 모델 구축
        self.integrated_model = self._build_integrated_model()

    def _load_sentence_type_model(self):
        """문장 유형 분류 모델 로드"""
        model_path = self.config['model_paths']['sentence_type']
        # 모델 로드 로직 구현
        return None  # 임시 반환

    def _load_neologism_model(self):
        """신조어 인식 모델 로드"""
        model_path = self.config['model_paths']['neologism']
        # 모델 로드 로직 구현
        return None  # 임시 반환

    def _load_aspect_sentiment_model(self):
        """속성 감정 분석 모델 로드"""
        model_path = self.config['model_paths']['aspect_sentiment']
        # 모델 로드 로직 구현
        return None  # 임시 반환

    def _load_empathy_model(self):
        """공감 응답 생성 모델 로드"""
        model_path = self.config['model_paths']['empathy']
        # 모델 로드 로직 구현
        return None  # 임시 반환

    def _build_integrated_model(self) -> Model:
        """통합 모델 구축"""
        # 입력 레이어
        text_input = Input(shape=(self.config['max_seq_length'],), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.config['max_seq_length'],), dtype=tf.int32, name='attention_mask')

        # BERT 인코더
        bert_model = TFBertModel.from_pretrained('klue/bert-base')
        bert_output = bert_model([text_input, attention_mask])[0]
        cls_output = bert_output[:, 0, :]

        # 특성 추출 레이어
        sentence_type_features = Dense(64, activation='relu', name='sentence_type_features')(cls_output)
        neologism_features = Dense(64, activation='relu', name='neologism_features')(cls_output)
        aspect_sentiment_features = Dense(128, activation='relu', name='aspect_sentiment_features')(cls_output)

        # 특성 통합
        merged_features = Concatenate(axis=1)([
            sentence_type_features,
            neologism_features,
            aspect_sentiment_features
        ])

        # 위기 단계 분류
        x = Dense(128, activation='relu')(merged_features)
        x = Dropout(0.3)(x)
        crisis_level = Dense(5, activation='softmax', name='crisis_level')(x)

        # 공감 응답 유형 분류
        y = Dense(128, activation='relu')(merged_features)
        y = Dropout(0.3)(y)
        empathy_type = Dense(4, activation='softmax', name='empathy_type')(y)

        # 통합 모델
        model = Model(
            inputs=[text_input, attention_mask],
            outputs=[crisis_level, empathy_type]
        )

        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss={
                'crisis_level': 'categorical_crossentropy',
                'empathy_type': 'categorical_crossentropy'
            },
            metrics={
                'crisis_level': 'accuracy',
                'empathy_type': 'accuracy'
            }
        )

        return model

    def train(self, train_data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], 
              validation_data: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], 
              epochs: int = 10, 
              batch_size: int = 32):
        """
        모델 훈련

        Args:
            train_data: 훈련 데이터 (입력, 레이블)
            validation_data: 검증 데이터 (입력, 레이블)
            epochs: 훈련 에포크 수
            batch_size: 배치 크기
        """
        # 훈련 데이터 준비
        train_inputs, train_labels = train_data
        val_inputs, val_labels = validation_data

        # 모델 훈련
        history = self.integrated_model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self._get_callbacks()
        )

        return history

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """훈련 콜백 설정"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config['model_paths']['checkpoints'],
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config['model_paths']['logs']
            )
        ]
        return callbacks

    def predict(self, text: str) -> Dict[str, Any]:
        """
        위기 단계 및 공감 응답 유형 예측

        Args:
            text: 입력 텍스트

        Returns:
            예측 결과 (위기 단계, 공감 응답 유형, 추가 분석 정보)
        """
        # 토큰화
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_seq_length'],
            return_tensors='tf'
        )

        # 예측
        crisis_level_probs, empathy_type_probs = self.integrated_model.predict(
            [inputs['input_ids'], inputs['attention_mask']]
        )

        # 결과 해석
        crisis_levels = ['정상군', '관찰필요', '상담필요', '학대의심', '응급']
        empathy_types = ['조언', '격려', '위로', '동조']

        crisis_level = crisis_levels[np.argmax(crisis_level_probs[0])]
        empathy_type = empathy_types[np.argmax(empathy_type_probs[0])]

        # 추가 분석
        sentence_type = self.sentence_type_model.predict(text) if self.sentence_type_model else None
        neologisms = self.neologism_model.detect(text) if self.neologism_model else None
        aspect_sentiments = self.aspect_sentiment_model.analyze(text) if self.aspect_sentiment_model else None

        # 결과 조합
        result = {
            'crisis_level': crisis_level,
            'crisis_level_confidence': float(np.max(crisis_level_probs[0])),
            'recommended_empathy_type': empathy_type,
            'empathy_type_confidence': float(np.max(empathy_type_probs[0])),
            'sentence_type': sentence_type,
            'detected_neologisms': neologisms,
            'aspect_sentiments': aspect_sentiments,
            'analysis': {
                'crisis_level_distribution': {
                    level: float(prob) for level, prob in zip(crisis_levels, crisis_level_probs[0])
                },
                'empathy_type_distribution': {
                    type_: float(prob) for type_, prob in zip(empathy_types, empathy_type_probs[0])
                }
            }
        }

        return result

    def generate_empathetic_response(self, text: str) -> str:
        """
        공감적 응답 생성

        Args:
            text: 입력 텍스트

        Returns:
            생성된 공감 응답
        """
        # 분석 결과 얻기
        analysis = self.predict(text)

        # 공감 유형에 맞는 응답 생성
        empathy_type = analysis['recommended_empathy_type']
        response = self.empathy_model.generate_response(text, empathy_type) if self.empathy_model else None

        return response

    def save(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        self.integrated_model.save(path)

    def load(self, path: str):
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        self.integrated_model = tf.keras.models.load_model(path) 