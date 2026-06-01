# Hybrid Energy-Based and Normalizing Flow Models for Trustworthy LLM Output Evaluation

## Overview

Large Language Models (LLMs) are capable of generating highly fluent and coherent text, but they often produce hallucinations, inconsistencies, and overconfident incorrect responses. This project introduces a hybrid deep learning framework that combines **Energy-Based Models (EBMs)** and **Normalizing Flow Models (NFMs)** to evaluate the trustworthiness of LLM-generated outputs.

The proposed system assigns a quantitative trust score to each response by integrating discriminative confidence estimation from EBMs with probabilistic uncertainty estimation from Normalizing Flows. This enables reliable detection of hallucinated, low-confidence, and anomalous outputs.

---

## Problem Statement

Traditional evaluation metrics focus primarily on linguistic quality and similarity-based measures. However, they fail to capture:

* Hallucinated information
* Model uncertainty
* Logical inconsistencies
* Reliability of generated responses

This limitation poses significant risks in domains such as healthcare, education, legal assistance, and decision-support systems.

---

## Objectives

* Detect hallucinated LLM outputs
* Estimate uncertainty using probabilistic modeling
* Develop a trustworthy evaluation framework
* Combine discriminative and generative learning approaches
* Improve reliability assessment of AI-generated content

---

## Dataset

### TruthfulQA Dataset

The project uses the TruthfulQA dataset containing:

* Questions
* Correct Answers
* Incorrect Answers

Data Preparation:

* Positive Samples → Question + Correct Answer
* Negative Samples → Question + Incorrect Answer

---

## Methodology

### 1. Feature Extraction

* Model: BERT (bert-base-uncased)
* Input: Question + Answer Pair
* Output: CLS Token Embedding

The extracted embeddings are used as feature representations for downstream trustworthiness evaluation.

---

### 2. Energy-Based Model (EBM)

The EBM assigns an energy score to each input.

#### Interpretation

* Low Energy → Trustworthy Output
* High Energy → Hallucinated Output

#### Training Objective

Contrastive Learning:

L = E(x_positive) − E(x_negative)

The model learns to assign lower energy to truthful responses and higher energy to hallucinated responses.

---

### 3. Normalizing Flow Model (NF)

Two flow architectures were explored:

* RealNVP
* Masked Autoregressive Flow (MAF)

#### Purpose

* Learn probability distribution of truthful responses
* Estimate uncertainty through exact likelihood computation
* Detect anomalous outputs

#### Interpretation

* High Likelihood → Reliable Output
* Low Likelihood → Potential Hallucination

---

### 4. Hybrid Trust Score

The final trust score is computed as:

Trust Score = α(-E(x)) + (1 − α)log p(x)

where:

* E(x) = Energy score
* log p(x) = Likelihood score
* α = Fusion weight

---

### 5. Fusion Framework

The outputs from EBM and NF models are combined using a fusion strategy to improve hallucination detection performance.

Pipeline:

Input → BERT → Feature Extraction

├── Energy-Based Model → Energy Score

├── Normalizing Flow → Likelihood Score

└── Fusion Module → Trustworthiness Score

Final Output → Trustworthy / Hallucinated

---

## Experimental Results

| Model                | Accuracy   | F1 Score  | AUROC     |
| -------------------- | ---------- | --------- | --------- |
| EBM                  | 74.25%     | 0.818     | 0.861     |
| RealNVP              | 75.24%     | 0.833     | 0.777     |
| MAF                  | 72.77%     | 0.818     | 0.740     |
| Hybrid (RealNVP)     | 74.25%     | 0.818     | 0.861     |
| Hybrid (MAF)         | 74.25%     | 0.819     | 0.862     |
| Fusion (MAF)         | 80.36%     | 0.868     | 0.889     |
| **Fusion (RealNVP)** | **82.84%** | **0.887** | **0.885** |

---

## Key Achievements

* Hallucination Detection
* Trust Score Generation
* Uncertainty Quantification
* Hybrid Deep Learning Architecture
* Improved Reliability Assessment
* Strong Classification Performance

---

## Technology Stack

### Deep Learning

* PyTorch
* Transformers
* BERT

### Generative Modeling

* RealNVP
* Masked Autoregressive Flow (MAF)

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

---

## Project Structure

```text
├── data/
│   └── TruthfulQA.csv
│
├── notebooks/
│   └── Hybrid_LLM_Evaluation.ipynb
│
├── models/
│   ├── EBM
│   ├── RealNVP
│   └── MAF
│
├── results/
│   ├── confusion_matrices
│   ├── evaluation_metrics
│   └── visualizations
│
└── README.md
```

---



