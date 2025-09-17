"""
handwritten_recognition.py
Handwritten Character Recognition with CNN (MNIST & EMNIST)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("TensorFlow not found. Please install TensorFlow 2.x")
    sys.exit(1)

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    print("scikit-learn not found. Install with `pip install scikit-learn`")
    sys.exit(1)

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("tensorflow-datasets not found. Install with `pip install tensorflow-datasets`")
    sys.exit(1)


def create_dirs():
    for folder in ['data', 'models', 'outputs']:
        os.makedirs(folder, exist_ok=True)


def load_data(dataset="MNIST"):
    create_dirs()
    if dataset.upper() == "MNIST":
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10

        os.makedirs('data/mnist', exist_ok=True)
        np.save('data/mnist/x_train.npy', x_train)
        np.save('data/mnist/y_train.npy', y_train)
        np.save('data/mnist/x_test.npy', x_test)
        np.save('data/mnist/y_test.npy', y_test)

    elif dataset.upper() == "EMNIST":
        emnist_dir = os.path.join('data', 'emnist')
        os.makedirs(emnist_dir, exist_ok=True)

        ds_train, ds_test = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            as_supervised=True,
            data_dir=emnist_dir,
            download=True
        )

        x_train, y_train = [], []
        for img, label in tfds.as_numpy(ds_train):
            x_train.append(img)
            y_train.append(label - 1)
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_test, y_test = [], []
        for img, label in tfds.as_numpy(ds_test):
            x_test.append(img)
            y_test.append(label - 1)
        x_test, y_test = np.array(x_test), np.array(y_test)

        num_classes = 26

        np.save(os.path.join(emnist_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(emnist_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(emnist_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(emnist_dir, 'y_test.npy'), y_test)

    else:
        raise ValueError("Dataset must be 'MNIST' or 'EMNIST'")

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"{dataset} loaded: train={x_train.shape}, test={x_test.shape}")
    return x_train, y_train, x_test, y_test, num_classes


def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('outputs/training_metrics.png')
    plt.close()


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()


def main():
    dataset_choice = "MNIST"   # change to "EMNIST" for letters
    x_train, y_train, x_test, y_test, num_classes = load_data(dataset_choice)

    model = build_cnn(input_shape=x_train.shape[1:], num_classes=num_classes)
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    model.save(f'models/cnn_{dataset_choice.lower()}.h5')
    plot_training(history)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = model.predict(x_test)
    plot_confusion(y_test, y_pred)

    print("\nClassification Report:\n")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


if __name__ == "__main__":
    main()
