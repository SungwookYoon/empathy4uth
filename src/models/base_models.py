import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import math

class BaseModel(nn.Module):
    """모든 모델의 기본 클래스"""
    def __init__(self, model_name="klue/bert-base", dropout_rate=0.3):
        super(BaseModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # BERT 파인튜닝을 위한 설정
        # 처음 몇 개의 레이어는 고정하여 과적합 방지
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state, outputs.pooler_output

    def init_weights(self, module):
        """가중치 초기화 함수"""
        if isinstance(module, nn.Linear):
            # He 초기화 적용
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class CrisisClassifier(BaseModel):
    """위기 단계 분류 모델"""
    def __init__(self, model_name="klue/bert-base", num_classes=5, dropout_rate=0.3):
        super(CrisisClassifier, self).__init__(model_name, dropout_rate)
        hidden_size = self.bert.config.hidden_size
        
        # 다중 레이어 분류기로 변경하여 표현력 증가
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 가중치 초기화
        self.classifier.apply(self.init_weights)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        _, pooled_output = super().forward(input_ids, attention_mask, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class SentenceTypeClassifier(BaseModel):
    """문장 유형 분류 모델"""
    def __init__(self, model_name="klue/bert-base", num_classes=5, dropout_rate=0.3):
        super(SentenceTypeClassifier, self).__init__(model_name, dropout_rate)
        hidden_size = self.bert.config.hidden_size
        
        # 다중 레이어 분류기로 변경
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 가중치 초기화
        self.classifier.apply(self.init_weights)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        _, pooled_output = super().forward(input_ids, attention_mask, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class AspectSentimentClassifier(BaseModel):
    """속성 기반 감정 분석 모델"""
    def __init__(self, model_name="klue/bert-base", num_aspects=5, num_sentiments=3, dropout_rate=0.3):
        super(AspectSentimentClassifier, self).__init__(model_name, dropout_rate)
        hidden_size = self.bert.config.hidden_size
        
        # 다중 레이어 분류기로 변경
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.aspect_classifier = nn.Linear(hidden_size // 2, num_aspects)
        self.sentiment_classifier = nn.Linear(hidden_size // 2, num_sentiments)
        
        # 가중치 초기화
        self.shared_layer.apply(self.init_weights)
        self.aspect_classifier.apply(self.init_weights)
        self.sentiment_classifier.apply(self.init_weights)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        _, pooled_output = super().forward(input_ids, attention_mask, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        
        shared_features = self.shared_layer(pooled_output)
        aspect_logits = self.aspect_classifier(shared_features)
        sentiment_logits = self.sentiment_classifier(shared_features)
        
        return aspect_logits, sentiment_logits

class EmpathyResponseGenerator(BaseModel):
    """공감 응답 생성 모델 (간소화된 버전)"""
    def __init__(self, model_name="klue/bert-base", vocab_size=32000, dropout_rate=0.3):
        super(EmpathyResponseGenerator, self).__init__(model_name, dropout_rate)
        hidden_size = self.bert.config.hidden_size
        
        self.decoder = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
        # 가중치 초기화
        self.fc.apply(self.init_weights)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                decoder_input_ids=None, decoder_attention_mask=None):
        encoder_hidden_states, _ = super().forward(input_ids, attention_mask, token_type_ids)
        
        # 간소화된 디코더 (실제로는 더 복잡한 로직이 필요)
        if decoder_input_ids is not None:
            decoder_embeddings = self.bert.embeddings(decoder_input_ids)
            decoder_output = self.decoder(
                decoder_embeddings, 
                encoder_hidden_states,
                tgt_key_padding_mask=~decoder_attention_mask.bool() if decoder_attention_mask is not None else None
            )
            logits = self.fc(decoder_output)
            return logits
        else:
            return encoder_hidden_states 