import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# tf.debugging.set_log_device_placement(True)

def get_data_label(dataset):
    data = dataset.unbatch().map(lambda x, y:x)
    labels = dataset.unbatch().map(lambda x, y:y)
    return data, labels

def normalize(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

def save_image(img, label, file, idx):
    out_path = f"../tea_leaf_augmented/{label}/{file}_{idx}.jpg"
    tf.keras.utils.save_img(out_path, img)
    # image = img.numpy().astype("uint8")
    # image.Save(out_path)

def augment_dataset(img, label, count, file, seed=42):
    out_path = f"../tea_leaf_augmented/{label}/{file}_orig.jpg"
    tf.keras.utils.save_img(out_path, img)
    # image = img.numpy().astype("uint8")
    # image.Save(out_path)

    for idx in range(count):
        # new_img = tf.image.stateless_random_brightness(image=new_img, max_delta=0.5, seed=(seed, seed))
        # new_img = tf.image.stateless_random_contrast(image=new_img, lower=0.5, upper=1.5, seed=(seed, seed))
        # new_img = tf.image.stateless_random_flip_left_right(image=new_img, seed=(seed, seed))
        # new_img = tf.image.stateless_random_flip_up_down(image=new_img, seed=(seed, seed))
        new_img = tf.image.stateless_random_hue(image=img, max_delta=0.2, seed=(seed, seed))
        new_img = tf.image.stateless_random_saturation(image=new_img, lower=0.75, upper=1.25, seed=(seed, seed))

        flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
        translation = tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest")
        rotate = tf.keras.layers.RandomRotation(factor=[0, 1])
        zoom = tf.keras.layers.RandomZoom(height_factor=[-0.1, 0.25], width_factor=[-0.1, 0.25], fill_mode="nearest")
        brightness = tf.keras.layers.RandomBrightness(0.2)
        contrast = tf.keras.layers.RandomContrast(0.2)

        layers = tf.keras.Sequential([
            flip,
            translation,
            rotate,
            zoom,
            brightness,
            contrast
        ])

        new_img = layers(new_img)

        save_image(new_img, label, file, idx)

directory = '../tea_dataset_merged'
# directory = '../tea sickness dataset'

seed = 42
dataset = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    shuffle=False
)

label_names = dataset.class_names
file_paths = dataset.file_paths
data, labels = get_data_label(dataset)
labels = np.array(list(labels))
unique, counts = np.unique(labels, return_counts=True)
print(np.asarray((unique, counts)).T)
num_items = {}
labels_map = dict(zip(np.arange(len(label_names)), label_names))

for label in label_names:
    num_items[label] = len(os.listdir(os.path.join(directory, label)))

print(labels_map)
print(num_items)

total_wanted = 5000
for d, l, f in zip(data, labels, file_paths):
    f = ''.join(f.split("/")[-1].split(".")[:-1])
    label = labels_map[l]
    existing_count = num_items[label]
    count_per_img = total_wanted // existing_count
    print(f)
    print(l, label, count_per_img)
    augment_dataset(d, label, count_per_img, f)
