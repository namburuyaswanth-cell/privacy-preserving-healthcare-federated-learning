"""
Federated CNN demonstration based on the implementation chapter
of the supplied final-year project report.

This is NOT a full OSDSTran implementation. The report's implementation
chapter uses a CNN + HMAC verification + FedAvg workflow.
"""

import os
import pickle
import hmac
import hashlib

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
IMG_SIZE = 128
SECRET_KEY = os.environ.get("HMAC_SECRET_KEY", "").encode()

if not SECRET_KEY:
    raise RuntimeError(
        "Set HMAC_SECRET_KEY before running. "
        "Do not hard-code secrets in the repository."
    )


def generate_hmac(data, key):
    message = pickle.dumps(data)
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_hmac(data, received_hmac, key):
    expected = generate_hmac(data, key)
    return hmac.compare_digest(expected, received_hmac)


def load_data():
    data, labels = [], []

    for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        folder = os.path.join(BASE_DIR, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

        for img_name in os.listdir(folder)[:300]:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def federated_avg(weights):
    return [
        np.mean(layer_group, axis=0)
        for layer_group in zip(*weights)
    ]


def main():
    X, y = load_data()
    print("Dataset:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X1, X_temp, y1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42
    )
    X2, X3, y2, y3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    clients = [(X1, y1), (X2, y2), (X3, y3)]

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    global_model = create_model()
    history = []
    rounds = 3

    for round_index in range(rounds):
        print(f"\n--- Round {round_index + 1} ---")
        local_weights = []

        for client_index, (X_local, y_local) in enumerate(clients, start=1):
            model = create_model()
            model.set_weights(global_model.get_weights())

            model.fit(
                datagen.flow(X_local, y_local, batch_size=16),
                epochs=5,
                verbose=1,
            )

            weights = model.get_weights()
            signature = generate_hmac(weights, SECRET_KEY)
            local_weights.append((weights, signature))

            print(f"Client {client_index}: update signed.")

        verified_weights = []

        for weights, signature in local_weights:
            if verify_hmac(weights, signature, SECRET_KEY):
                verified_weights.append(weights)
            else:
                print("Tampered client detected!")

        if not verified_weights:
            raise RuntimeError("No valid client updates were received.")

        global_model.set_weights(federated_avg(verified_weights))
        loss, accuracy = global_model.evaluate(X_test, y_test, verbose=0)
        history.append(accuracy)
        print(f"Global accuracy: {accuracy:.4f} | loss: {loss:.4f}")

    final_loss, final_accuracy = global_model.evaluate(
        X_test, y_test, verbose=0
    )

    probabilities = global_model.predict(X_test, verbose=0)
    y_pred = (probabilities > 0.5).astype(int).ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n===== FINAL RESULTS =====")
    print(f"Final Accuracy : {final_accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1 Score       : {f1 * 100:.2f}%")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(history, marker="o")
    plt.title("Federated Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(results_dir, "federated_accuracy.png"))
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    main()
