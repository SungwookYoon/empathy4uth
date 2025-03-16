import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
import math

class BertLstmHybridModel(nn.Module):
    """
    BERT-LSTM 하이브리드 모델 - 가스라이팅 탐지를 위한 모델
    
    논문에서 설명한 대로 BERT의 문맥적 임베딩 능력과 LSTM의 순차적 패턴 캡처 능력을
    결합한 하이브리드 아키텍처를 구현합니다.
    """
    def __init__(self, 
                 bert_model_name="klue/bert-base", 
                 hidden_size=768, 
                 lstm_hidden_size=256, 
                 num_classes=2, 
                 dropout_rate=0.2,
                 num_emotions=7,
                 emotion_embedding_dim=64,
                 num_attention_heads=8):
        super(BertLstmHybridModel, self).__init__()
        
        # BERT 인코딩 레이어
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dropout = nn.Dropout(dropout_rate)
        
        # 감정 임베딩 레이어
        self.emotion_embedding = nn.Embedding(num_emotions + 1, emotion_embedding_dim)  # +1 for padding/unknown
        
        # 양방향 LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0,
            batch_first=True
        )
        
        # 계층적 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # 양방향이므로 2배
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 감정 통합 레이어 (Feature-level fusion)
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2 + emotion_embedding_dim, lstm_hidden_size * 2),
            nn.LayerNorm(lstm_hidden_size * 2),
            nn.Tanh()
        )
        
        # 최종 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """가중치 초기화 함수"""
        if isinstance(module, nn.Linear):
            # He 초기화 적용
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask, emotion_ids=None, token_type_ids=None):
        """
        Args:
            input_ids: 입력 텍스트의 토큰 ID
            attention_mask: 어텐션 마스크
            emotion_ids: 감정 태그 ID (없을 경우 None)
            token_type_ids: 토큰 타입 ID (BERT 세그먼트)
        """
        # 입력 형태 확인 및 처리
        if len(input_ids.shape) == 3:  # [batch_size, max_turns, seq_len]
            # 대화 형태의 입력 처리
            batch_size, max_turns, seq_len = input_ids.shape
            
            # 각 턴별로 처리
            turn_embeddings = []
            for turn in range(max_turns):
                # 1. BERT 인코딩
                bert_outputs = self.bert(
                    input_ids=input_ids[:, turn],
                    attention_mask=attention_mask[:, turn],
                    token_type_ids=token_type_ids[:, turn] if token_type_ids is not None else None
                )
                sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
                sequence_output = self.bert_dropout(sequence_output)
                
                # CLS 토큰 임베딩 추출 (문장 표현)
                cls_embedding = sequence_output[:, 0]  # [batch_size, hidden_size]
                turn_embeddings.append(cls_embedding)
            
            # 턴 임베딩을 시퀀스로 결합
            turn_embeddings = torch.stack(turn_embeddings, dim=1)  # [batch_size, max_turns, hidden_size]
            
            # 2. 양방향 LSTM 처리
            lstm_output, _ = self.lstm(turn_embeddings)  # [batch_size, max_turns, lstm_hidden_size*2]
            
        else:  # [batch_size, seq_len]
            # 단일 텍스트 입력 처리
            # 1. BERT 인코딩
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            sequence_output = self.bert_dropout(sequence_output)
            
            # 2. 양방향 LSTM 처리
            lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, lstm_hidden_size*2]
        
        # 3. 셀프 어텐션 메커니즘
        attn_output, _ = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output
        )
        
        # 4. 감정 통합 (있는 경우)
        if emotion_ids is not None:
            # 감정 임베딩
            if len(emotion_ids.shape) == 2:  # [batch_size, max_turns]
                emotion_embeddings = self.emotion_embedding(emotion_ids)  # [batch_size, max_turns, emotion_dim]
                
                # 감정 임베딩과 LSTM 출력 결합
                if len(attn_output.shape) == 3:  # [batch_size, seq_len/max_turns, lstm_hidden*2]
                    # 시퀀스 길이 맞추기
                    seq_len = attn_output.shape[1]
                    if seq_len != emotion_embeddings.shape[1]:
                        # 길이가 다른 경우, 감정 임베딩 조정
                        if seq_len < emotion_embeddings.shape[1]:
                            emotion_embeddings = emotion_embeddings[:, :seq_len, :]
                        else:
                            # 패딩 추가
                            padding = torch.zeros(
                                emotion_embeddings.shape[0], 
                                seq_len - emotion_embeddings.shape[1], 
                                emotion_embeddings.shape[2],
                                device=emotion_embeddings.device
                            )
                            emotion_embeddings = torch.cat([emotion_embeddings, padding], dim=1)
                    
                    # 각 위치에서 감정 임베딩 결합
                    combined = torch.cat([attn_output, emotion_embeddings], dim=2)
                    fused_output = self.fusion(combined)
                else:
                    # 단일 표현으로 처리
                    emotion_embedding_avg = emotion_embeddings.mean(dim=1)  # [batch_size, emotion_dim]
                    attn_output_avg = attn_output.mean(dim=1)  # [batch_size, lstm_hidden*2]
                    combined = torch.cat([attn_output_avg, emotion_embedding_avg], dim=1)
                    fused_output = self.fusion(combined)
            else:  # [batch_size]
                emotion_embeddings = self.emotion_embedding(emotion_ids)  # [batch_size, emotion_dim]
                
                # 감정 임베딩과 LSTM 출력 결합 (평균 풀링)
                attn_output_avg = attn_output.mean(dim=1)  # [batch_size, lstm_hidden*2]
                combined = torch.cat([attn_output_avg, emotion_embeddings], dim=1)
                fused_output = self.fusion(combined)
        else:
            # 감정 태그 없이 처리
            fused_output = attn_output.mean(dim=1)  # [batch_size, lstm_hidden*2]
        
        # 5. 분류
        if len(fused_output.shape) == 3:  # [batch_size, seq_len, dim]
            # 시퀀스 평균 풀링
            fused_output = fused_output.mean(dim=1)  # [batch_size, dim]
        
        logits = self.classifier(fused_output)
        
        return {
            'logits': logits,
            'embeddings': fused_output
        }

class GaslightingDetector(nn.Module):
    """
    가스라이팅 탐지 시스템 - 대화 컨텍스트와 감정 태그를 활용한 탐지 시스템
    
    논문에서 설명한 대로 대화 컨텍스트를 분석하고 감정 태그를 통합하여
    가스라이팅 패턴을 탐지하는 종합적인 시스템을 구현합니다.
    """
    def __init__(self, 
                 bert_model_name="klue/bert-base",
                 hidden_size=768,
                 lstm_hidden_size=256,
                 num_classes=2,
                 dropout_rate=0.2,
                 num_emotions=7,
                 emotion_embedding_dim=64,
                 num_attention_heads=8,
                 max_conversation_turns=10,
                 max_seq_length=128):
        super(GaslightingDetector, self).__init__()
        
        self.max_conversation_turns = max_conversation_turns
        self.max_seq_length = max_seq_length
        
        # 기본 BERT-LSTM 하이브리드 모델
        self.bert_lstm = BertLstmHybridModel(
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            num_emotions=num_emotions,
            emotion_embedding_dim=emotion_embedding_dim,
            num_attention_heads=num_attention_heads
        )
        
        # 대화 컨텍스트 인코더
        self.context_encoder = nn.GRU(
            input_size=lstm_hidden_size * 2,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0,
            batch_first=True
        )
        
        # 감정 시퀀스 분석기
        self.emotion_sequence_encoder = nn.GRU(
            input_size=emotion_embedding_dim,
            hidden_size=emotion_embedding_dim // 2,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0,
            batch_first=True
        )
        
        # 최종 분류기
        self.final_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2 + emotion_embedding_dim, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # 위험 평가 모듈
        self.risk_assessor = nn.Sequential(
            nn.Linear(lstm_hidden_size + num_classes, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화 함수"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                conversation_input_ids, 
                conversation_attention_masks, 
                emotion_sequence=None,
                token_type_ids=None):
        """
        Args:
            conversation_input_ids: 대화 턴별 입력 ID [batch_size, max_turns, max_seq_length]
            conversation_attention_masks: 대화 턴별 어텐션 마스크 [batch_size, max_turns, max_seq_length]
            emotion_sequence: 대화 턴별 감정 태그 시퀀스 [batch_size, max_turns]
            token_type_ids: 토큰 타입 ID (옵션)
        """
        batch_size, num_turns = conversation_input_ids.shape[:2]
        
        # 각 대화 턴에 대한 BERT-LSTM 처리
        turn_representations = []
        for turn_idx in range(num_turns):
            # 현재 턴의 입력 추출
            turn_input_ids = conversation_input_ids[:, turn_idx]
            turn_attention_mask = conversation_attention_masks[:, turn_idx]
            
            # 현재 턴의 감정 태그 (있는 경우)
            turn_emotion = None
            if emotion_sequence is not None:
                turn_emotion = emotion_sequence[:, turn_idx]
            
            # 턴별 토큰 타입 ID (있는 경우)
            turn_token_type_ids = None
            if token_type_ids is not None:
                turn_token_type_ids = token_type_ids[:, turn_idx]
            
            # 유효한 턴만 처리 (패딩된 턴 제외)
            valid_turn_mask = turn_attention_mask.sum(dim=1) > 0
            if not valid_turn_mask.any():
                continue
                
            # BERT-LSTM 모델로 현재 턴 처리
            with torch.no_grad():  # 메모리 효율성을 위해 그래디언트 계산 방지
                turn_outputs = self.bert_lstm(
                    turn_input_ids,
                    turn_attention_mask,
                    turn_emotion,
                    turn_token_type_ids
                )
            
            # 턴 표현 저장
            turn_representations.append(turn_outputs['pooled_output'])
        
        # 턴 표현을 시퀀스로 결합 [batch_size, num_valid_turns, lstm_hidden_size*2]
        if turn_representations:
            turn_sequence = torch.stack(turn_representations, dim=1)
        else:
            # 유효한 턴이 없는 경우 (모두 패딩)
            return {
                'gaslighting_logits': torch.zeros(batch_size, 2, device=conversation_input_ids.device),
                'risk_score': torch.zeros(batch_size, 1, device=conversation_input_ids.device)
            }
        
        # 대화 컨텍스트 인코딩
        context_output, _ = self.context_encoder(turn_sequence)
        context_representation = torch.mean(context_output, dim=1)  # [batch_size, lstm_hidden_size*2]
        
        # 감정 시퀀스 분석 (있는 경우)
        emotion_context = None
        if emotion_sequence is not None:
            # 감정 임베딩 시퀀스 생성
            emotion_embeddings = self.bert_lstm.emotion_embedding(emotion_sequence)  # [batch_size, max_turns, emotion_dim]
            
            # 감정 시퀀스 인코딩
            emotion_output, _ = self.emotion_sequence_encoder(emotion_embeddings)
            emotion_context = torch.mean(emotion_output, dim=1)  # [batch_size, emotion_dim]
            
            # 컨텍스트와 감정 정보 결합
            combined_representation = torch.cat([context_representation, emotion_context], dim=1)
        else:
            # 감정 정보 없이 컨텍스트만 사용
            # 감정 임베딩 차원에 맞는 0 텐서 생성
            emotion_context = torch.zeros(batch_size, self.bert_lstm.emotion_embedding.embedding_dim, 
                                         device=context_representation.device)
            combined_representation = torch.cat([context_representation, emotion_context], dim=1)
        
        # 가스라이팅 분류
        gaslighting_logits = self.final_classifier(combined_representation)
        
        # 위험 평가
        risk_features = torch.cat([
            context_representation[:, :self.bert_lstm.lstm.hidden_size],  # 컨텍스트 일부만 사용
            F.softmax(gaslighting_logits, dim=1)  # 분류 확률 추가
        ], dim=1)
        risk_score = self.risk_assessor(risk_features)
        
        return {
            'gaslighting_logits': gaslighting_logits,
            'risk_score': risk_score,
            'context_representation': context_representation,
            'emotion_context': emotion_context
        }

class InterventionSystem:
    """
    개입 시스템 - 가스라이팅 탐지 결과에 따른 개입 전략을 구현
    
    논문에서 설명한 대로 위험 수준에 따라 3단계 개입 전략을 제공합니다.
    """
    def __init__(self):
        # 위험 수준 임계값
        self.LOW_RISK_THRESHOLD = 0.5
        self.MEDIUM_RISK_THRESHOLD = 0.7
        self.HIGH_RISK_THRESHOLD = 0.9
        
        # 개입 메시지 템플릿
        self.intervention_templates = {
            'low_risk': {
                'alert': "이 대화에서 가스라이팅 패턴이 감지되었을 수 있습니다.",
                'explanation': "상대방이 당신의 경험이나 감정을 부정하는 표현을 사용했을 수 있습니다.",
                'actions': ["자세히 알아보기", "계속 모니터링하기", "무시하기"]
            },
            'medium_risk': {
                'alert': "이 대화에서 가스라이팅 패턴이 감지되었습니다.",
                'explanation': "상대방이 당신의 현실 인식을 왜곡하거나 감정을 조작하려는 시도가 감지되었습니다.",
                'actions': ["대처 전략 보기", "신뢰할 수 있는 사람에게 연결하기", "대화 일시 중지하기"]
            },
            'high_risk': {
                'alert': "이 대화에서 심각한 가스라이팅 패턴이 감지되었습니다.",
                'explanation': "상대방이 지속적으로 당신의 현실 인식을 왜곡하고 있습니다. 이는 정신적 조작의 형태일 수 있습니다.",
                'actions': ["대화 중단 제안", "대응 가이드 보기", "상담 리소스에 연결하기"]
            }
        }
        
        # 가스라이팅 유형별 설명
        self.gaslighting_type_explanations = {
            'reality_distortion': "현실 왜곡: 상대방이 당신의 기억이나 경험을 부정하거나 왜곡하고 있습니다.",
            'emotional_manipulation': "감정 조작: 상대방이 당신의 감정 반응이 부적절하거나 과장되었다고 주장하고 있습니다.",
            'blame_shifting': "책임 전가: 상대방이 자신의 행동에 대한 책임을 당신에게 전가하고 있습니다.",
            'isolation': "고립: 상대방이 당신을 다른 사람들로부터 고립시키려 하고 있습니다.",
            'gradual_intensity': "점진적 강도: 상대방의 조작이 시간이 지남에 따라 점점 강해지고 있습니다."
        }
    
    def determine_intervention(self, gaslighting_probability, risk_score, gaslighting_type=None, user_profile=None):
        """
        탐지 결과에 따른 개입 전략 결정
        
        Args:
            gaslighting_probability: 가스라이팅 확률 (0~1)
            risk_score: 위험 점수 (0~1)
            gaslighting_type: 가스라이팅 유형 (옵션)
            user_profile: 사용자 프로필 정보 (옵션)
            
        Returns:
            개입 전략 정보를 담은 딕셔너리
        """
        # 위험 수준 결정
        if risk_score >= self.HIGH_RISK_THRESHOLD:
            risk_level = 'high_risk'
        elif risk_score >= self.MEDIUM_RISK_THRESHOLD:
            risk_level = 'medium_risk'
        elif risk_score >= self.LOW_RISK_THRESHOLD:
            risk_level = 'low_risk'
        else:
            return None  # 개입 필요 없음
        
        # 기본 개입 전략 선택
        intervention = self.intervention_templates[risk_level].copy()
        
        # 가스라이팅 유형에 따른 설명 추가 (있는 경우)
        if gaslighting_type and gaslighting_type in self.gaslighting_type_explanations:
            intervention['type_explanation'] = self.gaslighting_type_explanations[gaslighting_type]
        
        # 사용자 프로필에 따른 개인화 (있는 경우)
        if user_profile:
            intervention = self._personalize_intervention(intervention, user_profile)
        
        # 최종 개입 정보 반환
        return {
            'risk_level': risk_level,
            'alert_message': intervention['alert'],
            'explanation': intervention['explanation'],
            'type_explanation': intervention.get('type_explanation', ''),
            'action_options': intervention['actions'],
            'gaslighting_probability': float(gaslighting_probability),
            'risk_score': float(risk_score)
        }
    
    def _personalize_intervention(self, intervention, user_profile):
        """사용자 프로필에 따른 개입 전략 개인화"""
        personalized_intervention = intervention.copy()
        
        # 연령에 따른 조정
        if 'age' in user_profile:
            age = user_profile['age']
            if age < 15:  # 초기 청소년
                # 더 간단한 언어 사용
                personalized_intervention['alert'] = self._simplify_language(intervention['alert'])
                personalized_intervention['explanation'] = self._simplify_language(intervention['explanation'])
            elif age > 18:  # 후기 청소년
                # 더 상세한 정보 제공
                personalized_intervention['additional_info'] = "가스라이팅은 장기적으로 자존감과 정신 건강에 영향을 줄 수 있습니다."
        
        # 이전 경험에 따른 조정
        if 'previous_experiences' in user_profile and user_profile['previous_experiences'] > 0:
            # 이전에 가스라이팅을 경험한 사용자에게는 더 구체적인 전략 제공
            personalized_intervention['advanced_strategies'] = True
        
        return personalized_intervention
    
    def _simplify_language(self, text):
        """청소년 연령에 맞게 언어 단순화"""
        # 실제 구현에서는 더 복잡한 NLP 기반 단순화가 필요할 수 있음
        simplifications = {
            "가스라이팅": "마음 조작",
            "현실 인식을 왜곡": "생각을 헷갈리게 함",
            "정신적 조작": "마음을 혼란스럽게 하는 행동"
        }
        
        simplified_text = text
        for complex_term, simple_term in simplifications.items():
            simplified_text = simplified_text.replace(complex_term, simple_term)
            
        return simplified_text 