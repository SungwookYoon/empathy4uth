import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Any

class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        평가 지표 계산

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            y_proba: 예측 확률 (선택 사항)

        Returns:
            평가 지표 결과
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }

        # ROC-AUC (필요한 경우)
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')

        return metrics

    @staticmethod
    def calculate_crisis_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        위기 단계 평가 지표 계산

        Args:
            y_true: 실제 위기 단계
            y_pred: 예측 위기 단계
            y_proba: 예측 확률 (선택 사항)

        Returns:
            위기 단계 평가 지표
        """
        # 기본 지표
        metrics = EvaluationMetrics.calculate_metrics(y_true, y_pred, y_proba)

        # 위기 단계별 지표
        class_names = ['정상군', '관찰필요', '상담필요', '학대의심', '응급']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['class_report'] = report

        # 혼동 행렬
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # 심각한 위기 탐지 성능 (학대의심, 응급 등급)
        severe_indices = [3, 4]  # 학대의심, 응급 인덱스
        severe_true = np.isin(y_true, severe_indices).astype(int)
        severe_pred = np.isin(y_pred, severe_indices).astype(int)

        metrics['severe_crisis_precision'] = precision_score(severe_true, severe_pred)
        metrics['severe_crisis_recall'] = recall_score(severe_true, severe_pred)
        metrics['severe_crisis_f1'] = f1_score(severe_true, severe_pred)

        return metrics

    @staticmethod
    def calculate_empathy_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        공감 응답 평가 지표 계산

        Args:
            y_true: 실제 공감 유형
            y_pred: 예측 공감 유형
            y_proba: 예측 확률 (선택 사항)

        Returns:
            공감 응답 평가 지표
        """
        # 기본 지표
        metrics = EvaluationMetrics.calculate_metrics(y_true, y_pred, y_proba)

        # 공감 유형별 지표
        class_names = ['조언', '격려', '위로', '동조']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['class_report'] = report

        # 혼동 행렬
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    @staticmethod
    def calculate_aspect_sentiment_metrics(y_true: np.ndarray, y_pred: np.ndarray, aspects: List[str]) -> Dict[str, Any]:
        """
        속성별 감정 분석 평가 지표 계산

        Args:
            y_true: 실제 감정 레이블
            y_pred: 예측 감정 레이블
            aspects: 속성 목록

        Returns:
            속성별 감정 분석 평가 지표
        """
        metrics = {}

        # 속성별 평가
        for aspect in aspects:
            aspect_metrics = {
                'precision': precision_score(y_true[aspect], y_pred[aspect], average='weighted'),
                'recall': recall_score(y_true[aspect], y_pred[aspect], average='weighted'),
                'f1': f1_score(y_true[aspect], y_pred[aspect], average='weighted')
            }
            metrics[aspect] = aspect_metrics

        # 전체 평가
        metrics['overall'] = {
            'precision_macro': np.mean([m['precision'] for m in metrics.values()]),
            'recall_macro': np.mean([m['recall'] for m in metrics.values()]),
            'f1_macro': np.mean([m['f1'] for m in metrics.values()])
        }

        return metrics

    @staticmethod
    def calculate_neologism_metrics(true_neologisms: List[str], pred_neologisms: List[str]) -> Dict[str, float]:
        """
        신조어 인식 평가 지표 계산

        Args:
            true_neologisms: 실제 신조어 목록
            pred_neologisms: 예측된 신조어 목록

        Returns:
            신조어 인식 평가 지표
        """
        true_set = set(true_neologisms)
        pred_set = set(pred_neologisms)

        # 정확도 지표 계산
        true_positives = len(true_set.intersection(pred_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    @staticmethod
    def calculate_response_metrics(generated_responses: List[str], reference_responses: List[str]) -> Dict[str, float]:
        """
        생성된 응답 평가 지표 계산

        Args:
            generated_responses: 생성된 응답 목록
            reference_responses: 참조 응답 목록

        Returns:
            응답 평가 지표
        """
        # 응답 평가 지표 (BLEU, ROUGE 등) 구현
        # 현재는 더미 구현
        return {
            'bleu': 0.0,
            'rouge': 0.0
        } 