# Churn Prediction System

Teo's classes pushed me to build something more complete than a simple model — an end-to-end Machine Learning project with MLOps practices. The goal was to predict customer churn using real-life data from Teo's Twitch followers, and along the way, apply concepts that I had only seen in theory until then.

This project helped me evolve my understanding of Machine Learning pipelines and how to bring a model from training all the way to deployment.

Check the app:
[Streamlit](https://churnmlprediction-8r5awpjdlqrzcxdp8reqoj.streamlit.app/)

### Skills Applied

- Feature Engineering with Decision Tree-based discretization and One-Hot Encoding
- Model Training with Random Forest and GridSearch hyperparameter optimization
- Experiment Tracking with MLflow for model versioning and metrics logging
- Interactive Dashboard development with Streamlit
- Batch and individual prediction pipelines
- Model monitoring across Train, Test, and OOT datasets

## Project Overview

The dataset, provided during the course, contains real-life subscription data from Teo's community. The objective was to identify which customers were likely to churn, allowing for targeted retention strategies.

### Feature Engineering Pipeline

The pipeline was designed to handle raw data and transform it into meaningful features for the model. The key steps were decision tree discretization, one-hot encoding, and finally Random Forest classification.

```python
pipeline = Pipeline([
    ('discretizer', DecisionTreeDiscretiser(cv=3)),
    ('encoder', OneHotEncoder()),
    ('model', RandomForestClassifier())
])
```

To find the best model configuration, I used `GridSearchCV` to optimize the key hyperparameters:

```python
param_grid = {
    'model__min_samples_leaf': [15, 20, 25, 30, 50],
    'model__n_estimators': [100, 200, 500, 1000],
    'model__criterion': ['gini', 'entropy', 'log_loss']
}
```

### Experiment Tracking with MLflow

One of the main goals of this project was to practice MLOps concepts. Using MLflow, I was able to track each experiment, log metrics, and register the best model version — making it easy to compare runs and reproduce results.

```python
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("auc_roc_train", auc_train)
    mlflow.log_metric("auc_roc_test", auc_test)
    mlflow.sklearn.log_model(model, "random_forest_churn")
```

## Dashboard Development

After training and registering the model, I built a Streamlit application to make predictions accessible and interactive. The app supports two main workflows:

- **Batch Prediction**: Upload a CSV with customer data and download results with risk categorization and churn probabilities for the entire base.
- **Individual Prediction**: Input a single customer's features and get a real-time churn probability with a risk level assessment.

To run the application locally:

```bash
streamlit run app/streamlit_app.py
```

## Model Performance

| Dataset | Accuracy | AUC-ROC |
|---------|----------|---------|
| Train   | 0.761     | 0.848    |
| Test    | 0.756     | 0.826    |
| OOT     |-.---      | 0.836    |

The model showed stable performance across different datasets, indicating good generalization. On the training set, it achieved an AUC-ROC of 0.848 with 0.761 accuracy, demonstrating strong discriminative power.

Performance remained consistent on the test set, with an AUC-ROC of 0.826 and accuracy of 0.756, suggesting limited overfitting. On the Out-of-Time (OOT) dataset, the model reached an AUC-ROC of 0.836, reinforcing its robustness on unseen, time-shifted data.

Overall, the results indicate that the model is reliable and suitable for real-world churn prediction scenarios.

## Key Insights

- **Feature Engineering Impact**: Decision tree discretization significantly improved model performance by capturing non-linear relationships in the data.
- **Hyperparameter Optimization**: GridSearch revealed that larger forests (`n_estimators = 500+`) with higher `min_samples_leaf` values produced the most stable results across train and OOT datasets, reducing overfitting.
- **MLOps Value**: Tracking experiments with MLflow made it clear how much model performance can vary across runs — a practice that's easy to skip but hard to go back to ignoring once adopted.

This project was an important step in understanding not just how to train a model, but how to build something that can actually be maintained, versioned, and deployed in a real-world setting.
