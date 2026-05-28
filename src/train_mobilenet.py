import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout, BatchNormalization, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURASI DIREKTORI & PARAMETER
# ==========================================
# Skrip ini diasumsikan dijalankan dari root direktori proyek (pokemon-tcg-classification)
TRAIN_DIR = os.path.join('data', 'processed', 'train')
VAL_DIR = os.path.join('data', 'processed', 'val')
TEST_DIR = os.path.join('data', 'processed', 'test')

IMG_HEIGHT = 588
IMG_WIDTH = 420
BATCH_SIZE = 16 
NUM_CLASSES = 10

def main():
    print("=== Memulai Pipa Pelatihan MobileNetV2 (Micro-RoI) ===")
    
    # ==========================================
    # 2. PERSIAPAN DATA & AUGMENTASI
    # ==========================================
    print("\nMenyiapkan Data Generator...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        brightness_range=[0.8, 1.2]
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # Menghitung bobot kelas
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_generator.classes), 
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ==========================================
    # 3. PEMBANGUNAN ARSITEKTUR MODEL
    # ==========================================
    print("\nMembangun Arsitektur Micro-RoI...")
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Pemotongan ke kuadran ujung kiri bawah
    top_crop = int(IMG_HEIGHT * 0.85)
    bottom_crop = 0
    left_crop = 0
    right_crop = int(IMG_WIDTH * 0.70)
    
    x = Cropping2D(cropping=((top_crop, bottom_crop), (left_crop, right_crop)), name='micro_roi_cropping')(inputs)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    base_model.trainable = False 
    x = base_model(x)

    x = GlobalMaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    # ==========================================
    # 4. PELATIHAN TAHAP 1: FEATURE EXTRACTION
    # ==========================================
    print("\n[Tahap 1/2] Feature Extraction...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    model.fit(
        train_generator, 
        epochs=15, 
        validation_data=val_generator, 
        class_weight=class_weight_dict
    )

    # ==========================================
    # 5. PELATIHAN TAHAP 2: DEEP FINE-TUNING
    # ==========================================
    print("\n[Tahap 2/2] Deep Fine-Tuning (Unfreeze 50%)...")
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * 0.50)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    model.fit(
        train_generator, 
        epochs=40, 
        validation_data=val_generator, 
        callbacks=[early_stop, reduce_lr], 
        class_weight=class_weight_dict
    )

    # ==========================================
    # 6. EVALUASI DAN PENYIMPANAN
    # ==========================================
    print("\nMenyimpan Model...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'mobilenetv2_pokemon_tcg.keras')
    model.save(model_path)
    print(f"Model berhasil disimpan di: {model_path}")

    print("\nMengevaluasi model pada Test Set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Akurasi Akhir pada Test Set: {test_acc * 100:.2f}%\n")

    print("Membuat Laporan Klasifikasi...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

    # Menyimpan Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - MobileNetV2 (Micro-RoI)')
    plt.ylabel('Label Asli')
    plt.xlabel('Label Prediksi')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig(os.path.join('reports', 'confusion_matrix_final.png'))
    print("Confusion Matrix berhasil disimpan di folder 'reports/'.")
    print("=== Proses Selesai ===")

if __name__ == "__main__":
    main()