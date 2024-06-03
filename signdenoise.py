import numpy as np
import cv2
from PIL import Image
import os
import shutil
from keras import layers
from keras.models import Model

os.chdir('/Users/parva4/Documents/denoise')

def addRandomText(img_path):
    texts = ['Parva', 'Paul Walker', 'Smith', "Dwayne", "Tony Stark", 'Taylor',
    'Lewis', 'Hamilton', 'Max', "Vettori", 'Sergo Perez']
    font = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL]
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    noisy = image.copy()
    org = (w//2, h)
    fontScale = np.random.random() + 0.7
    thickness = np.random.randint (1, 3)
    noisy = cv2.putText(image, np.random.choice(texts), org, np.random.choice(font), fontScale, (0,0,0), thickness)
    if not os.path.exists('noise_cache'):
        os.mkdir('noise_cache')
    cv2.imwrite(os.path.join('noise_cache', 'noisy.jpg'), noisy)

def addRandomLines(index):
    image = cv2.imread(os.path.join('noise_cache', 'noisy.jpg'))
    noisy = image.copy()
    h, w, _ = image.shape
    num_lines = np.random.randint(2, 5)
    y_curr = int (h/num_lines)
    for i in range (num_lines):
        thickness = np.random.randint(1, 5)
        x1, x2 = 0, w
        y = y_curr * (i+1)
        noisy = cv2.line(noisy, (x1, y), (x2, y), color=(0, 0, 0), thickness=thickness)
        y_curr = y
    cv2.imwrite(os.path.join('Noised', f'noisy - {index}.jpg'), noisy)

def makeImgsNoisy(all_imgs_path=os.path.join('signatures', 'full_org')):
    if os.path.exists('Noised'):
        shutil.rmtree('Noised')
    os.mkdir('Noised')
    if os.path.exists('Output'):
        shutil.rmtree('Output')
    os.mkdir('Output')
    filenames = sorted(os.listdir(all_imgs_path))
    for i, name in enumerate(filenames):
        img_path = os.path.join(all_imgs_path, name)
        if img_path.endswith('.png') or img_path.endswith('.jpg'):
            shutil.copy(img_path, os.path.join('Output', f'clean-{i}.jpg'))
            addRandomText(img_path)
            addRandomLines(i)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L') 
    img = img.resize((300, 300))  # Adjusted to the desired input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    return img_array

def combineImgArrs(imgs_dir):
    img_paths = [os.path.join(imgs_dir, filename) for filename in sorted(os.listdir(imgs_dir))]
    img_arrs = [load_and_preprocess_image(img_path) for img_path in img_paths]
    return np.array(img_arrs)

def train_test_split():
    x_array, y_array = combineImgArrs('Noised'), combineImgArrs('Output')
    split_idx = int(0.8 * len(x_array))
    return x_array[:split_idx], x_array[split_idx:], y_array[:split_idx], y_array[split_idx:]

x_train, x_test, y_train, y_test = train_test_split()
input = layers.Input(shape=(300, 300, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

autoencoder.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, y_test),
)

# Save the trained autoencoder model
autoencoder.save('autoencoder_model.h5')
