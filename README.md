# 🎮 Dynamic Difficulty Adjustment (DDA) in Games using Machine Learning

This project demonstrates a machine learning-based Dynamic Difficulty Adjustment (DDA) system that classifies game difficulty (Easy, Medium, Hard) based on simulated player behavior data. It uses a Random Forest Classifier to model difficulty patterns, evaluate performance, and visualize decision-making logic.

---

## 📌 Project Overview

- **Purpose:** Automatically adjust game difficulty based on player behavior.
- **Model:** Random Forest Classifier.
- **Labels:** Easy, Medium, Hard (assigned using custom logic).
- **Dataset:** Synthetic gameplay data with realistic metrics.
- **Visualization:** Confusion Matrix Heatmap, Per-Class Recall, and Feature Importance.

---

## 💡 Features

- Simulates 500 player sessions with key features:
  - Reaction Time
  - Accuracy
  - Time Spent
  - Enemies Defeated
  - Final Score
- Custom rule-based logic to assign difficulty labels.
- Label encoding for supervised classification.
- Model training using stratified data split.
- Evaluation using classification report and confusion matrix.
- Graphical insight into model behavior and accuracy.

---

## 🛠 Technologies Used

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

---

## 📊 Model Evaluation

- Trained on 80% of the dataset, tested on 20%.
- Metrics include precision, recall, F1-score, and accuracy.
- Feature importance highlights which player behaviors influence difficulty predictions.

---

## 📁 Files Included

| File Name              | Description                                     |
|------------------------|-------------------------------------------------|
| `DDA_model.py`         | Full Python implementation of the model         |
| `README.txt`            | Project documentation (this file)               |
| `output_sample.png`    | Screenshot of model output/terminal results     |
| `confusion_matrix.png` | Heatmap visualization (auto-generated)          |
| `accuracy_chart.png`   | Per-class recall bar chart (auto-generated)     |

---

## 🔍 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SaadAyazCS/Dynamic-Difficulty-Adjustment.git
    cd Dynamic-Difficulty-Adjustment
