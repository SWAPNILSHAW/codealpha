# Creditworthiness Classification Project

## Overview
This project predicts creditworthiness using machine learning models on the UCI Default of Credit Card Clients dataset. It automates data download, preprocessing, feature engineering, model training, evaluation, and saves the best model and feature importances.

## Features
- Automatic dataset download (UCI Default of Credit Card Clients)
- Data preprocessing and feature engineering
- Trains Logistic Regression, Decision Tree, and Random Forest models
- Model evaluation (Precision, Recall, F1, ROC-AUC)
- Saves best model and feature importances
- Generates ROC curve and feature importance plots

## Project Structure
```
requirements.txt
src/
    creditworthiness_project.py
data/
model/
output/
```

## Setup
1. Clone the repository or copy the project files.
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```powershell
python src/creditworthiness_project.py
```

## Outputs
- Trained model: `model/best_model.pkl`
- Feature importances: `output/feature_importances.csv`, `output/feature_importances.png`
- ROC curve: `output/roc_curve.png`

## Customization
- To use a different dataset, update the `DATA_PATH` variable in `src/creditworthiness_project.py`.

## Requirements
- Python 3.7+
- See `requirements.txt` for required packages

## License
This project is for internship and educational purposes.
