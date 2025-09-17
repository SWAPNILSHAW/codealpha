# Handwritten Character Recognition using CNN

This project implements a Convolutional Neural Network (CNN) for handwritten character recognition using the MNIST and EMNIST datasets. The model is built with TensorFlow/Keras and provides training, evaluation, and visualization of results.

## Features
- Supports both MNIST (digits) and EMNIST (letters) datasets
- Automatic data download and preprocessing
- CNN architecture for high accuracy
- Training and validation accuracy/loss plots
- Confusion matrix and classification report
- Model saving for future inference

## Project Structure
```
handwritten_recognition/
├── data/           # Saved datasets
├── models/         # Saved models
├── outputs/        # Plots and results
├── src/
│   └── handwritten_recognition.py  # Main script
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- matplotlib
- seaborn
- tensorflow-datasets

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python src/handwritten_recognition.py
```

To switch between MNIST and EMNIST, change the `dataset_choice` variable in `handwritten_recognition.py`.

## Output
- Trained model saved in `models/`
- Training/validation accuracy and loss plot in `outputs/training_loss_accuracy.png`
- Confusion matrix in `outputs/confusion_matrix.png`
- Classification report printed to console

## License
This project is for educational and internship purposes.
