from keras_preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from train import train
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    # 训练
    model = train(
        "C:/Users/17173/Desktop/image_train_data/train",
        "C:/Users/17173/Desktop/image_train_data/model/model.h5"
    )
    # 校验
    model = load_model("C:/Users/17173/Desktop/image_train_data/model/model.h5")
    img = image.load_img(
        'C:/Users/17173/Desktop/image_train_data/valid/porn/[www.google.com][16973].jpg',
        target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0  # 与训练一致
    arr = np.expand_dims(arr, axis=0)
    predicted = model.predict(arr)
    predicted_class_indices = np.argmax(predicted, axis=1)
    labels = {'neutral': 0, 'political': 1, 'porn': 2, 'terrorism': 3}
    labels = dict((v, k) for k, v in labels.items())
    print("predicted_class :", labels[predicted_class_indices[0]])


if __name__ == '__main__':
    main()
