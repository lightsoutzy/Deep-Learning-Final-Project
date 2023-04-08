import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# tf.debugging.set_log_device_placement(True)

# utlities for visualizing images
def get_data_label(dataset):
    data = dataset.unbatch().map(lambda x, y:x)
    labels = dataset.unbatch().map(lambda x, y:y)
    return data, labels

# utility to inspect images
def visualize_dataset(training_data, augmentation=None):
    imgs, labels = get_data_label(training_data)
    imgs = list(imgs)
    print(np.min(imgs[0]), np.max(imgs[0]))
    labels = list(labels)
    labels_map = {
        0: 'algal leaf',
        1: 'Anthracnose',
        2: 'bird eye spot',
        3: 'brown blight',
        4: 'gray light',
        5: 'healthy',
        6: 'helopeltis',
        7: 'red leaf spot',
        8: 'white spot'
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = np.random.randint(len(training_data), size=(1,)).item()
        if augmentation:
            img = augmentation(imgs)[sample_idx].numpy().astype("uint8")
        else:
            img = imgs[sample_idx].numpy().astype("uint8")
        label = labels[sample_idx].numpy().item()
        print(label, np.min(img), np.max(img))
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()

directory = '../tea_dataset_merged'

# define hyperparameters
batch_size = 128
image_size = (256, 256)
seed = 42
epochs = 10

# load training and validation dataset from directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=0.3,
    subset='training',
    crop_to_aspect_ratio=False,
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=0.3,
    subset='validation',
    crop_to_aspect_ratio=False,
)
# 1/3 * 0.3 for test, 2/3 * 0.3 for validation
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take((1*val_batches) // 3)
val_dataset = val_dataset.skip((1*val_batches) // 3)

# normalize image data to be 0-1
def normalize(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

# normalize all train, validation, and test datasets
train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

def build_vggl16_raw():
    pre_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_vgg19_raw():
    pre_model = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_resnet50():
    pre_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_mobilenetv2_raw():
    pre_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_mobilenetv2():
    pre_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.MaxPool2D()(pre_model.output)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_inception_resnetv2():
    pre_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def build_efficient_net():
    pre_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    pre_model.trainable = False
    x = tf.keras.layers.Flatten()(pre_model.output)
    x = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = tf.keras.Model(inputs=pre_model.input, outputs=x)
    model.summary()
    return model

def train(model, train_ds, valid_ts):
    history = model.fit(train_ds,
        epochs=epochs,
        validation_data=valid_ts,
        verbose=2)

    return history

def test(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)
    print(test_loss, test_acc)

# model = build_vgg16_raw()
# model = build_vgg19_raw()
model = build_mobilenetv2()
# model = build_resnet50()
# model = build_inception_resnetv2()
# model = build_efficient_net()
# model = build_mobilenetv2_raw()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.008), metrics=['accuracy'])
history = train(model, train_dataset, val_dataset)
test(model, test_dataset)

