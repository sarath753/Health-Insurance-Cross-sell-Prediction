# 🚀 Health Insurance Cross-Sell Prediction  

## 📌 Project Overview  
This project focuses on predicting whether a customer will be interested in a vehicle insurance policy based on given demographic and historical data. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction).  

## 📁 Dataset  
- `train.csv`: Training dataset containing customer details and response labels.  
- `test.csv`: Test dataset used for predictions.  

## 🛠 Tech Stack  
- **Python**  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scikit-plot`, `xgboost`, `streamlit`  

## 📌 Features  
- Data preprocessing: Handling missing values, removing duplicates, and encoding categorical variables.  
- Data visualization: KDE plots, pie charts, and bar plots for understanding feature distributions.  
- Model training: Implemented Logistic Regression, SVM, Decision Trees, Random Forest, Extra Trees, and XGBoost.  
- Model evaluation: Confusion matrices, classification reports, and feature importance plots.  
- Model deployment: Streamlit-based interactive web app.  

## ⚙️ Installation  
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Create a Kaggle account and get an API token
- Go to Kaggle → Account → API.
- Click "Create New API Token" (this downloads a kaggle.json file).
- Keep this file ready to upload in the first code cell.

3. Run the Jupyter Notebook or Google Colab
- Open the notebook and upload your kaggle.json file when prompted.
- Run all the cells sequentially to download the dataset, train the model, and save the pipeline.

4. Launch the Streamlit app
- Once all cells are executed, start the Streamlit app by pasting the following link in your browser:
  ```
  https://a927-34-170-177-156.ngrok-free.app
  ```

## 🚀 Prediction  
To make predictions, use:  
```python
import joblib
import pandas as pd

pipeline = joblib.load('insurance_sell_prediction_pipeline.joblib')

sample_data = pd.DataFrame([{
    'Gender': 'Male',
    'Age': 30,
    'Driving_License': 1,
    'Region_Code': 28.0,
    'Previously_Insured': 0,
    'Vehicle_Age': '1-2 Year',
    'Vehicle_Damage': 'Yes',
    'Annual_Premium': 30000.0,
    'Policy_Sales_Channel': 26.0,
    'Vintage': 150
}])

pred = pipeline.predict(sample_data)
prob = pipeline.predict_proba(sample_data).max()
print(f'Prediction: {pred[0]}, Probability: {prob}')
```

## 📊 Model Performance  
| Model  | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|---------|
| Logistic Regression | 78.5% | 83.3% | 78.5% | 77.7% |
| Random Forest | 79.2% | 80.9% | 79.2% | 78.9% |
| XGBoost | 78.7% | 79.4% | 78.7% | 78.6% |

## 📌 Future Improvements  
- Hyperparameter tuning for better accuracy  
- Deploying the model as an API  
- Implementing deep learning approaches  

## 🤝 Contributing  
Feel free to open an issue or pull request for improvements!  
