import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import  ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l1,l2
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold

data_dir = 'D:\BaiduNetdiskDownload\甲状腺超声转移预测'
categories = ['有转移', '无转移']
img_size = (225, 225)
batch_size = 32

# load dataset
def load_data(data_dir, categories, img_size):
    data = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for root, dirs, files in os.walk(category_path):
            for file in files:
                img_path = os.path.join(root, file)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    data.append(img_array)
                    labels.append(class_num)
                except Exception as e:
                    print(e)

    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels


data, labels = load_data(data_dir, categories, img_size)



X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义 K 折交叉验证对象
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 准备用于存储交叉验证结果的列表
cross_val_scores = []

# 定义保存模型的路径
checkpoint_path = 'D:\PycharmProjects\MetastasisPre\\best_model.h5'

# 进行 K 折交叉验证
for fold_index, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
    X_train, X_val = data[train_index], data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    # model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)),
        # Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(lr=0.01)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 定义 ModelCheckpoint 回调函数，保存验证集准确率最高的模型
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',  # 监控验证集准确率
                                 save_best_only=True,  # 只保存表现最好的模型
                                 mode='max',  # 保存最大值的模型
                                 verbose=1)  # 显示保存信息

    # train
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
        epochs=30,
        callbacks=[TqdmCallback(), checkpoint],
        verbose=1
    )

    # 保存模型
    model.save(f'ultrasound_classification_model_{fold_index + 1}.h5')

    # 保存训练历史
    with open('train_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # 获取每次训练的历史记录
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # 绘制每次交叉验证的训练损失和验证损失变化图表
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Cross Validation Fold {fold_index + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'training_validation_loss_{fold_index + 1}.png')
    plt.show()

    # 绘制并保存训练和验证准确率变化图
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'training_validation_accuracy_{fold_index + 1}.png')  # 保存图表
    plt.show()

    # 在验证集上评估模型
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    cross_val_scores.append(accuracy)

    # 在测试集上评估最终选择的模型
    model = load_model(f'ultrasound_classification_model_{fold_index + 1}.h5')
    final_loss, final_accuracy = model.evaluate(X_test, y_test)
    print("Accuracy on test set:", final_accuracy)
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.8).astype("int32")

    print(classification_report(y_test, y_pred_classes, target_names=categories))

    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print(conf_matrix)

# 输出交叉验证的结果
print("Cross validation scores:", cross_val_scores)
print("Mean accuracy:", np.mean(cross_val_scores))

# 在测试集上评估最终选择的模型
model = load_model(checkpoint_path)
final_loss, final_accuracy = model.evaluate(X_test, y_test)
print("Final accuracy on test set:", final_accuracy)
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.8).astype("int32")

print(classification_report(y_test, y_pred_classes, target_names=categories))

conf_matrix = confusion_matrix(y_test, y_pred_classes)
print(conf_matrix)






