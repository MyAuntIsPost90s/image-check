import os

from PIL import ImageFile
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(data_dir, model_path):
    if not os.path.exists(data_dir):
        return
    # 创建数据集
    train_data = ImageDataGenerator(rescale=1. / 255,  # 预测的时候，特征也要这么处理
                                    shear_range=0.2,  # 用来进行剪切变换的程度
                                    zoom_range=0.2,  # 用来进行随机的放大
                                    horizontal_flip=True)  # 随机的对图片进行水平翻转
    train_generator = train_data.flow_from_directory(  # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签。
        data_dir,
        target_size=(224, 224),  # 所有图像将调整为224x224
        batch_size=32,
        class_mode='categorical'
    )
    print('数据集中的分类：' + str(train_generator.class_indices))

    print('正在创建训练模型...')
    if not os.path.exists(model_path):  # 从新创建模型
        # 创建 InceptionV3 训练模型
        pre_train_model = InceptionV3(weights='imagenet', include_top=False)
        # 添加全局平均池化层
        compute = GlobalAveragePooling2D()(pre_train_model.output)
        # 添加全连接
        compute = Dense(1024, activation='relu')(compute)
        # 添加一个分类器，由数据集得到分类数量
        predictions = Dense(len(train_generator.class_indices), activation='softmax')(compute)
        # 完成训练模型的创建
        train_model = Model(inputs=pre_train_model.input, outputs=predictions)
        # 冻结 InceptionV3 的卷积层提高训练速度
        for layer in pre_train_model.layers:
            layer.trainable = False
        # 编译模型
        train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
    else:  # 加载已有模型
        train_model = load_model(model_path)

    print("开始训练：")
    train_model.fit(
        train_generator,
        epochs=50
    )
    print("训练完毕")
    train_model.save(model_path)
    return train_model
