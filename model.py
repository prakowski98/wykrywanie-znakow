import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical

# Ścieżka do folderu archive
base_path = '/Users/prakowski_macbook/Desktop/wykrywanie-znakow-projekt/archive/'

# Funkcja do ładowania obrazów i etykiet z folderów
def load_images_from_folders(base_path):
    images = []
    labels = []
    for label in sorted(os.listdir(base_path)):  # Iteracja po folderach
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Nie można wczytać obrazu: {img_path}")
                continue
            img = cv2.resize(img, (32, 32))  # Skalowanie do 32x32 pikseli
            images.append(img)
            labels.append(int(label))  # Nazwa folderu to klasa
    return np.array(images), np.array(labels)

# Ładowanie danych treningowych
train_folder = os.path.join(base_path, 'Train')
X, y = load_images_from_folders(train_folder)

# Normalizacja obrazów
X = X / 255.0

# Podział na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding etykiet
y_train = to_categorical(y_train, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)

# Tworzenie modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacki: zapisywanie modelu i logowanie procesu trenowania
model_checkpoint = ModelCheckpoint('traffic_sign_model_epoch_{epoch:02d}.keras',
                                    save_best_only=True,
                                    monitor='val_accuracy',
                                    mode='max',
                                    verbose=1)
csv_logger = CSVLogger('training_log.csv', append=True)

# Trenowanie modelu
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint, csv_logger],
    verbose=1
)

# Załaduj plik Test.csv
test_csv_path = os.path.join(base_path, 'Test.csv')
test_data = pd.read_csv(test_csv_path)

# Funkcja do ładowania obrazów testowych na podstawie CSV
def load_test_images(data, folder_path):
    images = []
    labels = []
    for index, row in data.iterrows():
        # Poprawienie ścieżki do pliku: usunięcie prefiksu 'Test/' z Path
        relative_path = row['Path'].replace('Test/', '')  # Usunięcie zbędnego 'Test/'
        img_path = os.path.join(folder_path, relative_path)
        if not os.path.isfile(img_path):
            print(f"Plik nie istnieje: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Nie można wczytać obrazu: {img_path}")
            continue
        img = cv2.resize(img, (32, 32))  # Skalowanie do 32x32 pikseli
        images.append(img)
        labels.append(row['ClassId'])  # Klasa z CSV
    print(f"Wczytano {len(images)} obrazów z folderu testowego")
    return np.array(images), np.array(labels)

# Załaduj plik Test.csv
test_csv_path = os.path.join(base_path, 'Test.csv')
test_data = pd.read_csv(test_csv_path)

# Ścieżka do folderu testowego
test_folder = os.path.join(base_path, 'Test')

# Debugowanie kilku przykładowych ścieżek
print("Przykładowe ścieżki obrazów testowych:")
print(test_data.head())  # Sprawdź pierwsze 5 wierszy
print(f"Ścieżka do folderu testowego: {test_folder}")

# Ładowanie danych testowych
X_test, y_test = load_test_images(test_data, test_folder)

# Normalizacja danych testowych
X_test = X_test / 255.0

# One-hot encoding etykiet testowych
y_test = to_categorical(y_test, num_classes=43)

# Debugowanie danych testowych
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Ewaluacja modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")



# Zapis końcowego modelu w formacie .keras
model.save('traffic_sign_model_final.keras')
print("Końcowy model został zapisany jako 'traffic_sign_model_final.keras'.")
