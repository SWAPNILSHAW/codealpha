"""
handwritten_recognition.py
Project: Handwritten Character Recognition using CNN (MNIST & EMNIST)
Author: Generated for internship submission
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Safe Imports ------------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("Error: tensorflow.keras not found. Please install TensorFlow 2.x")
    sys.exit(1)

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    print("Error: scikit-learn not found. Install with `pip install scikit-learn`")
    sys.exit(1)

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("Error: tensorflow_datasets not found. Install with `pip install tensorflow-datasets`")
    sys.exit(1)

# ------------------------ Directory Setup ------------------------
def create_dirs():
    """Create required directories if they do not exist."""
    for folder in ['data', 'models', 'outputs']:
        if not os.path.exists(folder):
            os.makedirs(folder)

# ------------------------ Data Loading ------------------------
def load_data(dataset="MNIST"):
    """
    Load and preprocess dataset.
    Args:
        dataset: "MNIST" for digits, "EMNIST" for letters
    Returns:
        x_train, y_train, x_test, y_test: normalized and one-hot encoded
        num_classes: number of classes
    """
    create_dirs()
    
    if dataset.upper() == "MNIST":
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        
        # Save MNIST in data folder for future reference
        os.makedirs('data/mnist', exist_ok=True)
        np.save('data/mnist/x_train.npy', x_train)
        np.save('data/mnist/y_train.npy', y_train)
        np.save('data/mnist/x_test.npy', x_test)
        np.save('data/mnist/y_test.npy', y_test)
    
    elif dataset.upper() == "EMNIST":
        emnist_dir = os.path.join('data', 'emnist')
        os.makedirs(emnist_dir, exist_ok=True)
        
        # Load EMNIST letters
        ds_train, ds_test = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            as_supervised=True,
            data_dir=emnist_dir,
            download=True
        )
        
        # Convert to numpy arrays
        x_train, y_train = [], []
        for image, label in tfds.as_numpy(ds_train):
            x_train.append(image)
            y_train.append(label-1)  # EMNIST letters labels start at 1
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        x_test, y_test = [], []
        for image, label in tfds.as_numpy(ds_test):
            x_test.append(image)
            y_test.append(label-1)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        num_classes = 26
        
        # Save EMNIST in data folder
        np.save(os.path.join(emnist_dir,'x_train.npy'), x_train)
        np.save(os.path.join(emnist_dir,'y_train.npy'), y_train)
        np.save(os.path.join(emnist_dir,'x_test.npy'), x_test)
        np.save(os.path.join(emnist_dir,'y_test.npy'), y_test)
        
    else:
        raise ValueError("Dataset must be 'MNIST' or 'EMNIST'")
    
    # Normalize images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Expand dims for grayscale channel
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"{dataset} dataset loaded: x_train={x_train.shape}, x_test={x_test.shape}")
    return x_train, y_train, x_test, y_test, num_classes

# ------------------------ Model Building ------------------------
def build_cnn(input_shape, num_classes):
    """Build CNN model."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ------------------------ Training Plot ------------------------
def plot_training(history):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/training_loss_accuracy.png')
    plt.close()

# ------------------------ Confusion Matrix ------------------------
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

# ------------------------ Main Function ------------------------
def main():
    # Choose dataset
    dataset_choice = "MNIST"  # Change to "EMNIST" if you want letters
    
    x_train, y_train, x_test, y_test, num_classes = load_data(dataset_choice)
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=num_classes)
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )
    
    # Save model
    model.save(f'models/cnn_{dataset_choice.lower()}.h5')
    
    # Plot training
    plot_training(history)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = model.predict(x_test)
    plot_confusion(y_test, y_pred)
    
    print("\nClassification Report:\n")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

if __name__ == "__main__":
    main()
