# импорт необходимых библиотек
import streamlit as st
import numpy as np
import os
import pandas as pd
import warnings
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
import missingno as msgn
from sklearn import linear_model
import joblib
import librosa
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import argparse
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

from model.ReadData import ReadData
from model.Page import Page

readData = ReadData('voice.csv')
page = Page(readData)

# разделим данные с использованием Repeated Stratified K-Fold
X = readData.get_dataframe().drop('label', axis=1)
y = readData.get_dataframe()[['label']]
rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
lst_accu_stratified = []
for train_index, test_index in rskf.split(X, y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# модель логистической регрессии

# функция для модели логистической регрессии
def log_reg_with_repeat_fold(data, model):
    X = data.drop('label', axis=1)
    y = data[['label']]
    rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
    lst_accu_stratified = []
    for train_index, test_index in rskf.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        lst_accu_stratified.append(model.score(x_test, y_test))
    # сохраним модель
    filename = 'log_reg.pkl'
    joblib.dump(model, filename)
    return model, lst_accu_stratified, x_train, x_test


# загружаем модель логистической регрессии
log_reg, lst_accu_stratified, x_train, x_test = log_reg_with_repeat_fold(data=readData.get_dataframe(),
                                                                         model=linear_model.LogisticRegression())


def page11():
    max_accuracy = max(lst_accu_stratified) * 100
    min_accuracy = min(lst_accu_stratified) * 100
    overall_accuracy = np.mean(lst_accu_stratified) * 100
    std_deviation = np.std(lst_accu_stratified)
    st.title("Logistic Regression Model Training Results")
    st.write("Model training results:")
    st.write("Maximum Accuracy: {} %".format(max_accuracy))
    st.write("Minimum Accuracy: {} %".format(min_accuracy))
    st.write("Overall Accuracy: {} %".format(overall_accuracy))
    st.write("Standard Deviation: {}".format(std_deviation))
    st.write("\n*Train and Test sets are split")
    st.write("Train data shape:{}".format(x_train.shape))
    st.write("Test data shape:{}".format(x_test.shape))


# нейронная сеть

# переменная input_shape
input_shape = 20
input_shape = (X.shape[1],)


# определим функцию create_model, которая создает нейронную сеть
def create_model(input_shape):
    model = Sequential()
    # входной слой
    model.add(Dense(256, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.3))
    # скрытые слои
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # выходной слой
    model.add(Dense(1, activation="sigmoid"))
    # компиляция модели
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


# переменная model
model = create_model(input_shape)


# класс callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.94):
            print("\nReached 94% accuracy so cancelling training!")
            self.model.stop_training = True


# переменная callbacks
callbacks = myCallback()

# обучаем модель нейронной сети на train наборе данных
# определим batch_size и количество эпох обучения
batch_size = 64
epochs = 100
# обучаем модель на train наборе данных
model.fit(x_train, y_train, epochs=epochs,
          batch_size=batch_size,
          callbacks=[callbacks])

# сохраним модель
model.save('model.h5')

# загружаем веса
model.load_weights('model.h5')


# оценка производительности нейронной модели на test наборе данных
def page12():
    st.title('page12: oценка производительности нейронной модели на test наборе данных')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"Evaluating the neural network model using {len(x_test)} samples...")
    st.write(f"Loss: {loss:.4f}")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")


# цикл для предсказаний
preds = []
for i in range(0, len(x_test)):
    preds.append(model.predict(x_test)[i][0])
predictions = [1 if val > 0.5 else 0 for val in preds]


# результат бинарных предсказаний
def page13():
    st.title('page13: результат бинарных предсказаний')
    # точность бинарных предсказаний
    st.write("Overall Accuracy Score is : {}".format(accuracy_score(y_test, predictions)))


# кривая ROC и площадь под этой кривой (AUC) для нейронной модели
y_pred_keras = model.predict(x_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

# кривая ROC и площадь под этой кривой (AUC) для логистической модели
y_pred_lr = log_reg.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, thresholds_rf = roc_curve(y_test, y_pred_lr)
auc_lr = auc(fpr_lr, tpr_lr)


def page14():
    st.title('page14: визуализация производительности двух моделей')

    # График ROC curve
    plt.figure(1, figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.3f})'.format(auc_lr))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    st.pyplot(plt)

    # График ROC curve (zoomed in at top left)
    plt.figure(2, figsize=(8, 6))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.3f})'.format(auc_lr))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    st.pyplot(plt)


# функция визуализации предсказаний модели и распределения вероятностей для положительных и отрицательных классов
def plot_pdf(y_pred, y_test, name=None, smooth=500):
    positives = y_pred[y_test.label == 1]
    negatives = y_pred[y_test.label == 0]
    N = positives.shape[0]
    n = 10
    s = positives
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0]) / 2
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))

    N = negatives.shape[0]
    n = 10
    s = negatives
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0]) / 2
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))
    plt.xlim([0.0, 1.0])
    plt.xlabel('density')
    plt.ylabel('density')
    plt.title('PDF-{}'.format(name))
    plt.show()
    st.pyplot(plt)


def page15():
    st.title('page15: графики плотности вероятности для двух моделей')
    plot_pdf(y_pred_keras, y_test, 'Keras')
    plot_pdf(y_pred_lr, y_test, 'LR')


def model_testing():
    st.title('testing models: тестирование моделей')
    uploaded_file = st.file_uploader("загрузите файл данных (CSV)", type=["csv"])
    model_type = st.radio("выберите тип модели", ["логистическая регрессия", "нейронная сеть"])
    num_epochs = st.slider("количество эпох обучения (только для нейронной сети)", 1, 100, 10)

    if st.button('выполнить тестирование'):
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if model_type == "логистическая регрессия":
                page11()
            elif model_type == "нейронная сеть":
                page12()
                page13()


# выбор страницы
page = st.sidebar.selectbox("Выберите страницу", ["page 1", "page 2", "page 3",
                                                  "page 4", "page 5", "page 6",
                                                  "page 7", "page 8", "page 9",
                                                  "page 10", "page 11", "page 12",
                                                  "page 13", "page 14", "page 15", "model_testing"])

page_list = ["page 1", "page 2", "page 3",
             "page 4", "page 5", "page 6",
             "page 7", "page 8", "page 9",
             "page 10", "page 11", "page 12",
             "page 13", "page 14", "page 15", "model_testing"]

# отображение выбранной страницы
if page == "page 1":
    page.page1()
elif page == "page 2":
    page.page2()
elif page == "page 3":
    page.page3()
elif page == "page 4":
    page.page4()
elif page == "page 5":
    page.page5()
elif page == "page 6":
    page.page6()
elif page == "page 7":
    page.page7()
elif page == "page 8":
    page.page8()
elif page == "page 9":
    page.page9()
elif page == "page 10":
    page.page10()
elif page == "page 11":
    page11()
elif page == "page 12":
    page12()
elif page == "page 13":
    page13()
elif page == "page 14":
    page14()
elif page == "page 15":
    page15()
