import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    GlobalMaxPooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURASI DIREKTORI & PARAMETER
# ==========================================
TRAIN_DIR = os.path.join("data", "patches", "train")
VAL_DIR = os.path.join("data", "patches", "val")
TEST_DIR = os.path.join("data", "patches", "test")

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 10


def main():
    print("=== Memulai Pelatihan Patch-Based MobileNetV2 ===")

    # ==========================================
    # 2. PERSIAPAN DATA (NATIVE PREPROCESSING)
    # ==========================================
    print("\nMenyiapkan Data Generator...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        brightness_range=[0.8, 1.2],
        rotation_range=5,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ==========================================
    # 3. ARSITEKTUR PATCH-BASED (TANPA CROPPING INTERNAL)
    # ==========================================
    print("\nMembangun Arsitektur...")
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model(inputs)
    x = GlobalMaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(
        256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)

    # ==========================================
    # 4. TAHAP 1: FEATURE EXTRACTION
    # ==========================================
    print("\n[Tahap 1/2] Feature Extraction...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_phase1 = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        class_weight=class_weight_dict,
    )

    # ==========================================
    # 5. TAHAP 2: DEEP FINE-TUNING
    # ==========================================
    print("\n[Tahap 2/2] Deep Fine-Tuning (Unfreeze 50%)...")
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * 0.50)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    history_phase2 = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict,
    )

    # ==========================================
    # 6. MENYIMPAN MODEL
    # ==========================================
    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "mobilenetv2_pokemon_tcg.keras"))
    print("\nModel berhasil disimpan!")

    # ==========================================
    # 7. EVALUASI DAN VISUALISASI HASIL (TEST SET)
    # ==========================================
    print("\n=== Mengevaluasi Model pada Test Set ===")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Akurasi Akhir pada Test Set: {test_acc * 100:.2f}%\n")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

    # Pastikan folder reports ada
    os.makedirs("reports", exist_ok=True)

    # A. Membuat dan Menyimpan Confusion Matrix
    print("Menghasilkan dan menyimpan Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Patch-Based MobileNetV2")
    plt.ylabel("Label Asli")
    plt.xlabel("Label Prediksi")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "confusion_matrix_mobilenet.png"))
    plt.close()

    # B. Membuat dan Menyimpan Learning Curves
    print("Menghasilkan dan menyimpan Learning Curves...")
    acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
    val_acc = (
        history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
    )
    loss = history_phase1.history["loss"] + history_phase2.history["loss"]
    val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]

    initial_epochs = len(history_phase1.history["accuracy"])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Grafik Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Akurasi Latih")
    plt.plot(epochs_range, val_acc, label="Akurasi Validasi")
    plt.axvline(
        x=initial_epochs, color="green", linestyle="--", label="Mulai Fine-Tuning"
    )
    plt.title("Kurva Akurasi")
    plt.xlabel("Epochs")
    plt.ylabel("Akurasi")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Grafik Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Loss Latih")
    plt.plot(epochs_range, val_loss, label="Loss Validasi")
    plt.axvline(
        x=initial_epochs, color="green", linestyle="--", label="Mulai Fine-Tuning"
    )
    plt.title("Kurva Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.savefig(os.path.join("reports", "learning_curves_mobilenet.png"))
    plt.close()

    print("=== Seluruh Laporan Berhasil Disimpan di Direktori 'reports/' ===")


if __name__ == "__main__":
    main()
