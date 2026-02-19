
# AI600 – Assignment 1  
### Neural Networks, Backpropagation & Bias Gradient Analysis

---

## 📌 Overview

This repository contains the complete implementation and analysis for **AI600 – Assignment 1**, including:

- Exploratory Data Analysis (EDA)
- NumPy-based MLP (from scratch)
- Gradient magnitude tracking
- Gradient-based feature attribution
- PyTorch MLP implementation
- Shared vs Independent Bias gradient derivations (Q2)
- Numerical gradient checking
- Convergence and optimisation comparison

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-MLP%20from%20Scratch-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![LaTeX](https://img.shields.io/badge/LaTeX-Analytical%20Derivations-green.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

---

# 📂 Project Structure

data/               → Training & test datasets  
outputs/            → Metrics & CSV outputs  
reports/plots/      → Generated visualisations  
src/                → Source code  
&nbsp;&nbsp;&nbsp;&nbsp;q1/             → Question 1 implementation  
&nbsp;&nbsp;&nbsp;&nbsp;q2/             → Question 2 implementation  

---

# 🚀 How to Run

## Question 1

### 1️⃣ Part A – Exploratory Data Analysis

```
python -m src.q1.partA_eda     --train data/train.csv     --out reports/plots
```

---

### 2️⃣ Part B(a) + B(b) – NumPy MLP (Sigmoid + ReLU + Gradient Tracking)

```
python -m src.q1.partB_train_numpy     --train data/train.csv     --out outputs     --plots reports/plots     --iters 250     --lr 0.01
```

---

### 3️⃣ Part C(a) – Analytical Gradient-Based Feature Attribution

See the submitted PDF report for full derivations.

---

### 4️⃣ Part C(b) + Part D – PyTorch MLP + Gradient Attribution + Test Evaluation

```
python -m src.train_torch     --train data/train.csv     --test data/test.csv     --out outputs     --plots reports/plots     --epochs 25     --lr 1e-3
```

---

# Question 2  
## Bias Gradients, Parameter Sharing & Convergence Analysis

Run the complete Q2 experiment (gradient checks + training + plots):

```
python -m src.q2.experiments.run_experiment
```

This will:

- Compute analytical shared-bias gradients
- Perform numerical gradient checking
- Train shared-bias and independent-bias models
- Generate accuracy and loss comparison plots

---

# 📊 Outputs

Generated artifacts include:

- Accuracy curves (NumPy & PyTorch)
- Gradient magnitude plots
- Feature attribution rankings
- Shared vs Independent bias comparison plots
- CSV metric files

---

# 📄 Report

The full analytical derivations and experimental discussion are included in the submitted PDF report.

---

## 👤 Author

- Junaid Tariq
- MS AI 2025-27
- AI600 – Neural Networks & Deep Learning University Submission Repository Academic Use Only
