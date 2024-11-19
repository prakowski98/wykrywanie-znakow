import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tensorflow.keras.models import load_model

# Ścieżka do zapisanych modeli i danych
base_path = '/Users/prakowski_macbook/Desktop/wykrywanie-znakow-projekt/archive/'
model_path = 'traffic_sign_model_final.keras'
test_csv_path = os.path.join(base_path, 'Test.csv')
test_folder = os.path.join(base_path, 'Test')

# Przetłumaczone nazwy klas
classes = {
    0: 'Ograniczenie prędkości (20km/h)', 
    1: 'Ograniczenie prędkości (30km/h)', 
    2: 'Ograniczenie prędkości (50km/h)', 
    3: 'Ograniczenie prędkości (60km/h)',
    4: 'Ograniczenie prędkości (70km/h)', 
    5: 'Ograniczenie prędkości (80km/h)', 
    6: 'Koniec ograniczenia prędkości (80km/h)', 
    7: 'Ograniczenie prędkości (100km/h)',
    8: 'Ograniczenie prędkości (120km/h)', 
    9: 'Zakaz wyprzedzania', 
    10: 'Zakaz wyprzedzania dla pojazdów powyżej 3.5 t', 
    11: 'Pierwszeństwo na skrzyżowaniu',
    12: 'Droga z pierwszeństwem', 
    13: 'Ustąp pierwszeństwa', 
    14: 'Stop', 
    15: 'Zakaz ruchu pojazdów', 
    16: 'Zakaz ruchu pojazdów powyżej 3.5 t', 
    17: 'Zakaz wjazdu',
    18: 'Ogólny znak ostrzegawczy', 
    19: 'Niebezpieczny zakręt w lewo', 
    20: 'Niebezpieczny zakręt w prawo', 
    21: 'Podwójny zakręt',
    22: 'Wyboista droga', 
    23: 'Śliska nawierzchnia', 
    24: 'Zwężenie drogi po prawej stronie', 
    25: 'Roboty drogowe', 
    26: 'Sygnalizacja świetlna', 
    27: 'Przejście dla pieszych',
    28: 'Przejście dla dzieci', 
    29: 'Przejście dla rowerzystów', 
    30: 'Uwaga: oblodzenie/śnieg', 
    31: 'Przejście dla dzikich zwierząt',
    32: 'Koniec ograniczeń prędkości i zakazów wyprzedzania', 
    33: 'Nakaz skrętu w prawo', 
    34: 'Nakaz skrętu w lewo', 
    35: 'Nakaz jazdy prosto',
    36: 'Nakaz jazdy prosto lub w prawo', 
    37: 'Nakaz jazdy prosto lub w lewo', 
    38: 'Nakaz omijania przeszkody z prawej strony', 
    39: 'Nakaz omijania przeszkody z lewej strony', 
    40: 'Ruch okrężny',
    41: 'Koniec zakazu wyprzedzania', 
    42: 'Koniec zakazu wyprzedzania dla pojazdów powyżej 3.5 t'
}

# Funkcja do wczytywania danych testowych na podstawie CSV
def load_test_images(data, folder_path):
    images = []
    labels = []
    paths = []
    for index, row in data.iterrows():
        relative_path = row['Path'].replace('Test/', '')  # Usuń 'Test/' z Path
        img_path = os.path.join(folder_path, relative_path)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (32, 32)) / 255.0  # Normalizacja
        images.append(img)
        labels.append(row['ClassId'])
        paths.append(img_path)
    return np.array(images), np.array(labels), paths

# Załaduj plik Test.csv
test_data = pd.read_csv(test_csv_path)

# Wczytaj dane testowe
X_test, y_test, image_paths = load_test_images(test_data, test_folder)

# Wczytaj zapisany model
model = load_model(model_path)

# Klasa do interaktywnego przeglądania zdjęć
class ImageViewer:
    def __init__(self, model, X_test, y_test, image_paths):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.image_paths = image_paths
        self.index = 0
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Przycisk "Następny"
        self.ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.btn_next = Button(self.ax_next, 'Następny')
        self.btn_next.on_clicked(self.next_image)

        # Przycisk "Poprzedni"
        self.ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Poprzedni')
        self.btn_prev.on_clicked(self.prev_image)

        # Wyświetl pierwsze zdjęcie
        self.display_image()

    def display_image(self):
        self.ax.clear()
        img = self.X_test[self.index]
        true_label = self.y_test[self.index]
        pred_label = np.argmax(self.model.predict(img.reshape(1, 32, 32, 3)))

        # Oryginalne zdjęcie
        image_path = self.image_paths[self.index]
        img_original = cv2.imread(image_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)  # Konwersja na RGB

        # Wyświetlenie obrazu
        self.ax.imshow(img_original)
        self.ax.axis('off')
        self.ax.set_title(f"Prawdziwa klasa: {classes[true_label]}\n"
                           f"Predykcja: {classes[pred_label]}")
        self.fig.canvas.draw()

    def next_image(self, event):
        self.index = (self.index + 1) % len(self.X_test)
        self.display_image()

    def prev_image(self, event):
        self.index = (self.index - 1) % len(self.X_test)
        self.display_image()

# Uruchom przeglądarkę zdjęć
viewer = ImageViewer(model, X_test, y_test, image_paths)
plt.show()
