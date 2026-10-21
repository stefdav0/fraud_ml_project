# Fraud Detection Using Machine Learning

A machine learning project that explores the detection of fraudulent transactions using Python and Scikit-learn.

The project began as a tutorial-based implementation and is being extended through additional model evaluation, classification threshold analysis, and data visualization.

## Project Objectives

- Train a machine learning model to classify transactions as legitimate or fraudulent.
- Evaluate model performance using multiple classification metrics.
- Investigate the challenges of imbalanced classification.
- Compare classification thresholds and their impact on predictions.
- Visualize the trade-off between Precision, Recall, and F1-score.
- Develop the project further into an application with an API and frontend.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
fraud_ml_project/
├── analysis.ipynb
├── fraud_detection.py
├── fraud_detection_model.pkl
├── threshold_comparison.png
├── .gitignore
└── README.md
```

- `analysis.ipynb`: Data analysis, model training, evaluation, and threshold experiments.
- `fraud_detection.py`: Application code from the initial implementation.
- `fraud_detection_model.pkl`: Saved machine learning pipeline.
- `threshold_comparison.png`: Visualization of evaluation metrics across different thresholds.
- `.gitignore`: Files and directories excluded from Git tracking.

The dataset is stored locally and is not included in this repository.

## Initial Tutorial and Project Development

This project initially started as a learning exercise based on the following YouTube tutorial:

**Original tutorial:** [Fraud Detection Using Machine Learning – Full Python Data Science Project (94% Accuracy)](https://www.youtube.com/watch?v=4Od5_z28iIE)

The tutorial provided the initial learning foundation for exploring the dataset, preparing the data, training a classification model, and saving the trained pipeline.

Building on this initial implementation, the project is being extended with additional experiments, evaluation visualizations, and documentation.

The original tutorial is acknowledged as the starting point of the project. The additional analyses and planned improvements are documented separately to distinguish the initial implementation from subsequent development.

**Licensing note:** The original tutorial's code licensing terms have not yet been verified. Its public availability on YouTube does not automatically grant permission to redistribute or reuse the code. The original license should be checked before redistributing the original implementation.

## Our Extensions and Improvements

### 1. Model Evaluation

The model is evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted fraud cases that are actually fraudulent.
- **Recall:** Proportion of actual fraud cases detected.
- **F1-score:** Harmonic mean of Precision and Recall.

These metrics provide a more informative assessment than Accuracy alone when working with imbalanced data.

### 2. Confusion Matrix

A confusion matrix helps us understand the model's predictions through four categories:

- **True Positives (TP):** Fraudulent transactions correctly identified.
- **True Negatives (TN):** Legitimate transactions correctly identified.
- **False Positives (FP):** Legitimate transactions incorrectly flagged as fraudulent.
- **False Negatives (FN):** Fraudulent transactions incorrectly classified as legitimate.

### 3. Classification Threshold Analysis

We experimented with thresholds from 0.50 to 0.90 to investigate how changing the decision threshold affects the model's predictions.

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives |
| ---------: | --------: | -----: | -------: | --------------: | --------------: |
| 0.50 | 0.0225 | 0.9448 | 0.0439 | 101,241 | 136 |
| 0.70 | 0.0391 | 0.8981 | 0.0750 | 54,325 | 251 |
| 0.80 | 0.0540 | 0.8498 | 0.1015 | 36,697 | 370 |
| 0.90 | 0.0839 | 0.7881 | 0.1516 | 21,210 | 522 |

Among the tested thresholds, 0.90 achieved the highest F1-score. However, increasing the threshold also reduced Recall, meaning that more fraudulent transactions were missed.

The appropriate threshold depends on the balance between missed fraud cases and false alarms. Further validation is needed before selecting a final threshold. The comparisons presented here are exploratory and use the current test set.

### 4. Threshold Comparison Visualization

The following figure shows how Precision, Recall, and F1-score change as the classification threshold increases.

![Threshold Comparison](threshold_comparison.png)

The visualization highlights the trade-off between detecting more fraudulent transactions and reducing false positives.

## Current Model Performance

At the default threshold of 0.50, the model achieved the following results on the current test set:

| Metric | Result |
| ------ | -----: |
| Accuracy | 94.69% |
| Precision | 2.25% |
| Recall | 94.48% |
| F1-score | 0.0439 |

The model achieves high Recall but low Precision. This indicates that it detects most fraudulent transactions, while also incorrectly flagging many legitimate transactions.

Because the dataset is highly imbalanced, Accuracy alone is not sufficient to evaluate the model's usefulness.

These are experimental results, not evidence of production readiness.

## Future Work

- Evaluate threshold selection using a separate validation set.
- Investigate methods to reduce false positives.
- Extend model evaluation and error analysis.
- Develop a FastAPI backend for serving predictions.
- Build a frontend with an interactive threshold slider.
- Connect the frontend to the API.
- Add application screenshots and usage instructions.

## Learning Outcomes

This project provides practical experience with:

- Machine learning classification pipelines.
- Model evaluation on imbalanced datasets.
- Confusion matrices and classification metrics.
- Classification threshold experimentation.
- Saving and reusing trained models.
- Version control and project documentation with Git and GitHub.

## Disclaimer

This project is intended for educational purposes. Its performance depends on the dataset and evaluation methodology, and it has not been validated for real-world financial decisions.