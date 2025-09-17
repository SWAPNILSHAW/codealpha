# Disease Prediction Pipeline

This project provides a machine learning pipeline for disease prediction using three datasets: breast cancer, diabetes, and heart disease. It trains multiple models, evaluates their performance, and saves results and predictions for further analysis.

## Features
- Supports breast cancer, diabetes, and heart disease datasets
- Models: Logistic Regression, SVM, Random Forest, XGBoost
- Automatic preprocessing and feature scaling
- Hyperparameter tuning (optional)
- Saves predictions, results, confusion matrices, ROC curves, and best models
- Outputs results in Excel and PNG formats

## Project Structure
```
├── data/                # Input datasets (CSV)
├── models/              # Saved model files (.pkl)
├── outputs/             # Results, plots, and predictions
├── src/
│   └── disease_prediction.py  # Main pipeline script
├── requirements.txt     # Python dependencies
```

## Setup
1. Install Python 3.8+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place datasets in the `data/` folder (if not automatically downloaded)

## Usage
Run the pipeline from the project root:
```
python src/disease_prediction.py
```
To enable hyperparameter tuning:
```
python src/disease_prediction.py --tune
```

## Outputs
- Confusion matrices and ROC curves: `outputs/`
- Predictions and results: `outputs/` (Excel files)
- Best models: `models/`

## Requirements
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- openpyxl

## License
This project is for internal use and demonstration purposes.
