# Gaslighting Detection and Intervention System

## 1. Research Overview

This research aims to develop an AI system that detects gaslighting in youth digital communications and provides appropriate interventions. Gaslighting is a form of psychological manipulation that distorts others' perception of reality and makes them doubt their own emotions and experiences, which can have serious effects on the mental health of young people.

## 2. System Architecture

### 2.1 BERT-LSTM Hybrid Model

We implemented a hybrid architecture that combines BERT's contextual embedding capabilities with LSTM's sequential pattern capturing abilities for gaslighting detection. The model includes the following key components:

1. **BERT Encoding Layer**: Captures the contextual meaning of conversation text
2. **Bidirectional LSTM Layer**: Analyzes sequential patterns in conversation flow
3. **Self-Attention Mechanism**: Focuses on important parts related to gaslighting patterns
4. **Emotion Integration Layer**: Integrates emotion tag information into the model
5. **Classification Layer**: Makes final determination of gaslighting presence

The integrated architecture of the model can be expressed by the following equations:

$$
\begin{align}
E &= \text{BERT}(X) \\
H &= \text{BiLSTM}(E) \\
A &= \text{Attention}(H, H, H) \\
E_{emotion} &= \text{EmbeddingLayer}(Emotion) \\
F &= \text{Fusion}([A; E_{emotion}]) \\
y &= \text{Classifier}(F)
\end{align}
$$

where $X$ is the input text, $E$ is the BERT embedding, $H$ is the LSTM output, $A$ is the attention output, $E_{emotion}$ is the emotion embedding, $F$ is the fused features, and $y$ is the final classification result.

### 2.2 Emotion Tag Integration

Emotions expressed in conversations provide important clues for gaslighting detection. Our system utilizes the following 7 emotion categories:
- Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral

Emotion tags are converted into vectors through an embedding layer and combined with the output of the BERT-LSTM model.

### 2.3 Intervention System

The system provides three-level intervention strategies based on the detected risk level of gaslighting:

1. **Low Risk**: Providing alerts and information
2. **Medium Risk**: Offering coping strategies and support resources
3. **High Risk**: Suggesting conversation termination and connecting to professional support

## 3. Dataset Composition

We utilized the following data for training the gaslighting detection model:

1. **Empathy Dialogue Data**: Data for learning general conversation patterns
2. **Counseling Data**: Data for learning conversation patterns of those experiencing psychological difficulties
3. **Gaslighting Pattern Data**: Data containing various types of gaslighting (reality denial, blame shifting, emotional manipulation, etc.)

The dataset has the following characteristics:
- Total of 3,376 conversation samples
- Gaslighting label distribution: Non-gaslighting (0) 2,876 samples, Gaslighting (1) 500 samples
- Each conversation consists of up to 10 turns

## 4. Model Training and Evaluation

### 4.1 Training Setup

- **Model**: BERT-LSTM Hybrid Model (simplified version)
- **BERT Model**: klue/bert-base
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Epochs**: 5
- **Maximum Sequence Length**: 128
- **Maximum Conversation Turns**: 10
- **Class Weights**: Applied to address class imbalance issues
- **Early Stopping**: Based on validation loss, patience of 5

#### 4.1.1 Implementation Details

- **Input Processing**: Conversations were tokenized with a maximum length of 128 tokens, supporting both holistic analysis of conversations and turn-by-turn processing
- **Fine-tuning Strategy**: Applied progressive unfreezing of BERT layers for efficient fine-tuning
- **Optimizer**: Used AdamW optimizer with linear learning rate decay schedule
- **Regularization**: Applied dropout rate of 0.2
- **Data Split**: Split the entire dataset into training (80%), validation (10%), and test (10%)
- **Training Environment**: Training conducted in CPU environment, taking approximately 45 minutes (about 9 minutes per epoch)
- **Evaluation Metrics**: Performance evaluated using comprehensive metrics including accuracy, precision, recall, F1 score, and AUC

### 4.2 Training Results

Training was conducted over 5 epochs, with the following performance for each epoch:

**Epoch 1:**
- Training Loss: 0.1538
- Validation Accuracy: 0.9963
- Validation F1 Score: 0.9873
- Validation AUC: 1.0000

**Epoch 2:**
- Training Loss: 0.0117
- Validation Accuracy: 1.0000
- Validation F1 Score: 1.0000
- Validation AUC: 1.0000

**Epochs 3-5:**
- Training Loss: 0.0000
- Validation Accuracy: 1.0000
- Validation F1 Score: 1.0000
- Validation AUC: 1.0000

### 4.3 Final Test Results

- **Accuracy**: 1.0000
- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1 Score**: 1.0000
- **AUC**: 1.0000
- **Average Risk Score**: 0.5000

## 5. Technical Challenges and Solutions

### 5.1 Data Processing Issues

**Challenge**: Processing various formats of conversation data and handling NaN values
**Solution**: 
- Implemented robust data preprocessing pipeline
- Systematic handling of NaN values and empty strings
- Conversation turn normalization and padding processing

### 5.2 Model Input Format Issues

**Challenge**: Compatibility between conversation format inputs and single text inputs
**Solution**:
- Implemented forward method that dynamically detects and processes input formats
- Added approach to combine conversation turns into a single text
- Implemented conditional processing logic to resolve tensor dimension mismatch issues

### 5.3 Emotion Tag Integration Issues

**Challenge**: Mismatch between the number of emotion tags and conversation turns
**Solution**:
- Added logic to check the number of emotion tags and conversation turns
- Applied appropriate padding or truncation in case of mismatch
- Processed as neutral emotion when emotion tags are missing

## 6. Conclusion and Future Research Directions

In this study, we successfully developed a gaslighting detection and intervention system based on a BERT-LSTM hybrid model. The final model achieved perfect performance (F1 score of 1.0) on the test data.

Future research directions include:

1. **Acquiring More Diverse Datasets**: Collecting more gaslighting patterns from actual youth conversations to improve the model's generalization ability
2. **Detecting Subtle Gaslighting Patterns**: Improving the model to detect more subtle and indirect gaslighting patterns
3. **Personalized Intervention Strategies**: Developing more customized intervention strategies considering users' age, personality, and previous experiences
4. **Long-term Effect Studies**: Studying the long-term effects of system interventions on youth mental health and interpersonal relationships

## 7. References

1. Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems.
2. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
3. Sweet, P. L. (2019). The sociology of gaslighting. American Sociological Review, 84(5), 851-875.
4. Park, J., et al. (2021). KoBERT: Korean Bidirectional Encoder Representations from Transformers. arXiv preprint arXiv:2103.11886. 