from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.urls import path
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import handing_app.identify_method as hi
# 模型识别调用方法
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter




# 主页（显示界面）
def index(request):
    return render(request,'index.html')

# 文件上传页面
def upload_files(request):
    print("这是upload_files页面！！！！")
    return render(request,'upload_files.html')


# 检查上传分片是否重复，如果重复则不提交，否则提交
@csrf_exempt
def checkChunk(request):
    print("action checkChunk")
    # post请求
    if request.method == 'POST':
        # 获得上传文件块的大小,如果为0，就告诉他不要上传了
        chunkSize = request.POST.get("chunkSize")
        if chunkSize == '0':
            return JsonResponse({'ifExist': True})
        # 如果文件块大小不为0 ，那么就上传，需要拼接一个临时文件
        file_name = request.POST.get('fileMd5') + request.POST.get('chunk')

        # 如果说这个文件不在已经上传的目录，就可以上传，已经存在了就不需要上传。
        if file_name not in get_deep_data():
            return JsonResponse({'ifExist': False})
        return JsonResponse({'ifExist': True})


# 判断一个文件是否在一个目录下
def get_deep_data(path='upload/'):
    result = []
    data = os.listdir(path)
    for i in data:
        if os.path.isdir(i):
            get_deep_data(i)
        else:
            result.append(i)
    return result


# 前端上传的分片 保存到 指定的目录下
@csrf_exempt
def upload(request):
    print("action upload")
    if request.method == 'POST':
        md5 = request.POST.get("fileMd5")
        chunk_id = request.POST.get("chunk", "0")
        fileName = "%s-%s" % (md5, chunk_id)
        file = request.FILES.get("file")
        with open('upload/' + fileName, 'wb') as f:
            for i in file.chunks():
                f.write(i)
        return JsonResponse({'upload_part': True})



# 将每次上传的分片合并成一个新文件
@csrf_exempt
def mergeChunks(request):
    print("action mergeChunks")
    if request.method == 'POST':
        print("=====================================")
        # 获取需要给文件命名的名称
        fileName = request.POST.get("fileName")
        # 该图片上传使用的md5码值
        md5 = request.POST.get("fileMd5")
        id = request.POST.get("fileId")
        # 分片的序号
        chunk = 0
        # 完成的文件的地址为
        path = os.path.join('upload', fileName)
        with open(path, 'wb') as fp:
            while True:
                try:
                    filename = 'upload/{}-{}'.format(md5, chunk)
                    with open(filename, 'rb') as f:
                        fp.write(f.read())
                    # 当图片写入完成后，分片就没有意义了，删除
                    os.remove(filename)
                except:
                    break
                chunk += 1
        return JsonResponse({'upload': True, 'fileName': fileName, 'fileId': id})









# 模型识别页面
def identify_model(request):
    print("++++++++++++++++++++++++++++++++++")
    folder_path = "upload"
    file_names = os.listdir(folder_path)
    validate_files = ["train_features(MINE).csv", "train_label（MINE）.csv",
                      "test_features（MINE）.csv","test_label(MINE).csv"]
    # 判断文件是否存在？
    tag = True
    for f in file_names:
        if f not in validate_files:
            os.remove(f"upload/{f}")
            tag = False

    print("==================================")
    print(tag)

    if tag:
        chart_values = []
        acc_values = []

        MODELS_DIR = 'models/'
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        MODEL_TF = MODELS_DIR + 'model'
        MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
        MODEL_TFLITE = MODELS_DIR + 'model.tflite'
        MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

        data_train = pd.read_csv(r"upload\train_features(MINE).csv")
        data_labels = pd.read_csv(r"upload\train_label（MINE）.csv")
        data = pd.concat([data_train, data_labels], axis=1)
        data = shuffle(data)
        print(data.shape)

        x = data.drop('500', axis=1)
        y = data['500']
        print(data_train.shape, data_labels.shape)
        x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.20)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_validate = tf.keras.utils.to_categorical(y_validate)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_validate = scaler.transform(x_validate)

        test_data = pd.read_csv(r"upload\test_features（MINE）.csv")
        test_label = pd.read_csv(r"upload\test_label(MINE).csv")
        test_data = pd.concat([test_data, test_label], axis=1)
        x_test = test_data.drop('500', axis=1)
        y_test = test_data['500']
        x_test = scaler.transform(x_test)

        """#Create a model"""
        model = tf.keras.Sequential()
        model.add(layers.Dense(7, activation='relu', input_shape=(500,)))
        model.add(layers.Dense(4, activation='softmax'))
        # model.add(layers.Dropout(0.15))
        model.summary()

        """Fit the model"""
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        EPOCHS = 1000
        callbacks = [EarlyStopping(monitor='val_loss', patience=25),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                     save_best_only=True)]  # uses validation set to stop training when it start overfitting
        history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs=EPOCHS,
                            callbacks=callbacks,
                            batch_size=32, verbose=1, shuffle=True)

        model = load_model('./best_model.h5')
        model.save('best_model.h5')
        metrics = history.history

        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        # plt.show()
        plt.savefig('loss.png')
        chart_values.extend([history.epoch, metrics['loss'], metrics['val_loss']])

        """Check test accuracy"""

        model = load_model('best_model.h5')
        # model.save(MODEL_TF)
        y_pred = np.reshape(np.argmax(model.predict(x_test), axis=1), [len(x_test), 1])
        y_true = np.reshape(y_test.values, [-1, 1])

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm2 = cm.ravel().reshape(cm.shape).tolist()
        print(type(cm2), cm2)

        # 将混淆矩阵转换为[x, y, value]的格式
        data = []
        size = len(cm)
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                data.append([j, size - 1 - i, value])
        print("=============================================")
        print(data)

        # 散点图
        yp = np.unique(y_pred.flatten()).tolist()

        counter = dict(Counter(y_pred.flatten().tolist())).values()
        counter = [value for value in counter]

        test_acc = float(sum(y_pred == y_true) / len(y_true))
        print('Test accuracy is:')
        print(f"{test_acc:.2%}")
        model.save(MODEL_TF)
        acc_values.append(f"{test_acc:.2%}")

        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
        model_no_quant_tflite = converter.convert()

        # Save the model to disk
        open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

        def representative_dataset():
            # 50 -> 5
            for i in range(5):
                yield [x_train[i, :].reshape(1, 500).astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce integer only quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # Provide a representative dataset to ensure we quantize correctly.
        converter.representative_dataset = representative_dataset
        model_tflite = converter.convert()

        # Save the model to disk
        open(MODEL_TFLITE, "wb").write(model_tflite)

        (hi.predict_tflite(model_no_quant_tflite, x_test[1, :]))

        """Note depending on the model the quantized version might have higher accuracy. """

        # Calculate predictions with full software
        y_pred = np.reshape(np.argmax(model.predict(x_test), axis=1), [len(x_test), 1])
        test_acc = float(sum(y_pred == y_true) / len(y_true))
        print('Test accuracy with model:')
        print(f"{test_acc:.2%}")

        y_test_pred_no_quant_tflite = np.empty([x_test.shape[0], 1])
        y_test_pred_tflite = np.empty([x_test.shape[0], 1])
        # Calculate predictions with tensorflow lite
        for i in range(0, x_test.shape[0]):
            y_test_pred_no_quant_tflite[i, 0] = np.argmax(hi.predict_tflite(model_no_quant_tflite, x_test[i, :]))

        test_acc = float(sum(y_test_pred_no_quant_tflite == y_true) / len(y_true))
        print('Test accuracy with model tf lite:')
        print(f"{test_acc:.2%}")
        acc_values.append(f"{test_acc:.2%}")

        # Calculate predictions with tensorflow lite quantized model
        for i in range(0, x_test.shape[0]):
            y_test_pred_tflite[i, 0] = np.argmax(hi.predict_tflite(model_tflite, x_test[i, :]))

        test_acc = float(sum(y_test_pred_tflite == y_true) / len(y_true))
        print('Test accuracy with model quantized:')
        print(f"{test_acc:.2%}")
        acc_values.append(f"{test_acc:.2%}")

        y_test_tflite = {"y_test_pred_tflite": y_test_pred_tflite, "x_test": x_test}
        sio.savemat('models/tflite_pred.mat', y_test_tflite)

        # Calculate size
        size_no_quant_tflite = os.path.getsize(MODEL_NO_QUANT_TFLITE)
        size_tflite = os.path.getsize(MODEL_TFLITE)

        # Compare size
        # Compare size
        pd.DataFrame.from_records(
            [["TensorFlow Lite", f"{size_no_quant_tflite} bytes ", f"(reduced by {0} bytes)"],
             ["TensorFlow Lite Quantized", f"{size_tflite} bytes",
              f"(reduced by {size_no_quant_tflite - size_tflite} bytes)"]],
            columns=["Model", "Size", ""], index="Model")

        print(chart_values)

        return render(request, 'identify_model.html',
                      context={'chart_values': chart_values, 'acc_values': acc_values, 'data': data, "yp": yp,
                               "counter": counter})
    else:
        return render(request,'errors.html')





