# Financial-Sentiment-LoRA-Finetune

Fine-tuning **DistilBERT** with **LoRA (Low-Rank Adaptation)** for financial sentiment analysis using the **Financial PhraseBank** dataset.

---

##  Overview  
This project fine-tunes a lightweight transformer (**DistilBERT**) using **parameter-efficient fine-tuning (LoRA)** on the **Financial PhraseBank** dataset to classify financial news headlines into:
- **Positive**
- **Neutral**
- **Negative**

The dataset consists of short financial sentences labeled by experts.  
This model is designed to capture subtle sentiment signals in finance-related texts, enabling efficient sentiment scoring for investment, trading, and news analytics applications.

---

##  Key Features  
- Fine-tuning with **LoRA** for efficient adaptation  
- Custom **Weighted Loss** to address class imbalance  
- Detailed **EDA** (Exploratory Data Analysis) with visualizations  
- End-to-end training + evaluation + real-world financial examples  
- Clean, reproducible code using Hugging Face Transformers  

---

## Dataset Summary  
**Source:** [Financial PhraseBank - Sentences_AllAgree.txt](https://huggingface.co/datasets/hadyelsahar/financial_phrasebank)

| Sentiment | Samples | Percentage |
|------------|----------|-------------|
| Negative   | 303      | 13.6%       |
| Neutral    | 1386     | 62.3%       |
| Positive   | 570      | 24.1%       |

The dataset is moderately imbalanced, with a dominance of neutral sentences.

---

## Addressing Class Imbalance  
To prevent bias toward the majority (neutral) class, class imbalance was addressed using **class-weighted cross-entropy loss**.

w_i = N / (k × n_i)

where  
- \( N \) = total number of samples  
- \( k \) = number of classes  
- \( n_i \) = number of samples in class *i*

These weights are applied directly in the loss computation through a **custom `WeightedTrainer`** class extending Hugging Face’s `Trainer`.

---

##  Model Architecture  
- **Base Model:** `distilbert-base-uncased`  
- **Task Type:** Sequence Classification (3 labels)  
- **Parameter-Efficient Tuning:** LoRA  
  - Rank (`r`) = 8  
  - Alpha = 16  
  - Dropout = 0.1  
  - Target Modules: `["q_lin", "v_lin"]` (DistilBERT attention layers)

Only LoRA adapters are trained,the rest of the base model remains frozen, significantly reducing compute cost while retaining performance.

---

## Training Setup

| Component | Value |
|------------|--------|
| **Epochs** | 3 |
| **Batch Size (Train)** | 16 |
| **Batch Size (Eval)** | 32 |
| **Learning Rate** | 2e-4 |
| **Warmup Steps** | 100 |
| **Optimizer** | AdamW |
| **Evaluation Metric** | F1-macro |
| **Hardware** | CUDA-compatible GPU (recommended) |

---

## Metrics
After training, performance was evaluated across **train**, **validation**, and **test** sets using:
- **Accuracy**
- **F1-macro**
- **Confusion Matrix**
- **Classification Report**

The **macro-F1** ensures balanced performance across all sentiment classes, regardless of class frequency.

---

## Example Predictions
The model was tested on unseen financial headlines:

| Text | Predicted Sentiment | Confidence |
|------|----------------------|-------------|
| "The company's quarterly earnings exceeded expectations..." | **Positive** | 98% |
| "Revenue declined by 8% year-over-year..." | **Negative** | 94% |
| "The Federal Reserve maintained interest rates..." | **Neutral** | 91% |
| "Bankruptcy filing announced after failed restructuring..." | **Negative** | 96% |
| "Dividend payout increased by 20%..." | **Positive** | 95% |

---

##  Project Workflow

1. **Data Loading**  
   - Load `Sentences_AllAgree.txt`  
   - Parse sentences and labels using `@` delimiter  

2. **EDA (Exploratory Data Analysis)**  
   - Check for nulls, duplicates, and class distribution  
   - Visualize sentiment ratios using `matplotlib` and `seaborn`

3. **Preprocessing & Splitting**  
   - Encode sentiment labels (`negative`, `neutral`, `positive`)  
   - Split into train/val/test with 70/15/15 stratification  

4. **Model Setup & Tokenization**  
   - Load `DistilBERT` tokenizer  
   - Apply LoRA configuration  

5. **Training with Class Weights**  
   - Use a custom loss function for weighted class balancing  
   - Evaluate each epoch using F1-macro  

6. **Evaluation & Results**  
   - Generate confusion matrices & reports  
   - Save model and tokenizer to `./sentiment_lora_model`  

7. **Inference on Financial Texts**  
   - Predict sentiment and confidence scores for unseen sentences  

---
id2label = {0: "negative", 1: "neutral", 2: "positive"}
print(f"Sentiment: {id2label[prediction]}")
