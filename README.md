# Streamlit AVM Deployment
By Pari Katyal and Ryan Shihabi

## Models Deployed
1. XGBoost
2. Swin Transformer

## Dashboard Site
[*Dashboard*](http://13.58.45.37:8501
)

## Endpoints
EC2 Endpoint 
 - http://3.140.207.64:5000/predict

XGBoost Backup Endpoint
 - http://23.240.69.246/xgboost_predict

Hybrid CNN Backup Endpoint
 - http://23.240.69.246/image_predict

Setup Instructions:
1. Backup Endpoints (Optional):

`python3 api.py`
