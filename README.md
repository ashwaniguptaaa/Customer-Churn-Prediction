# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave the company.

## Live Demo

https://customer-churn-prediction-demwxtwflxr8qxxq8wg6ji.streamlit.app/

## Project Overview

Customer churn is an important business problem. This project uses customer information and service details to predict the likelihood of churn.

## Dataset

- Telco Customer Churn dataset
- 7,043 customer records
- 19 input features after removing the customer ID
- Target variable: Churn

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## Machine Learning Model

A tuned Decision Tree classification pipeline was trained to predict customer churn.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 74.59% |
| Precision | 51.40% |
| Recall | 78.34% |
| F1 Score | 62.08% |

These results are from the test dataset.

## Features

- Customer information input
- Churn prediction
- Churn probability
- Customer risk category
- Prediction recommendations
- Model performance dashboard
- Confusion matrix
- Feature importance visualization

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Disclaimer

This is a learning project. Predictions are estimates and should not be treated as guaranteed business outcomes.
