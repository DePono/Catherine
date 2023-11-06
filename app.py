# импорт необходимых библиотек
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn import linear_model

from model.ReadData import ReadData
from model.Page import Page
from model.LogRegression import LogRegression
from model.Prepare import Prepare
from model.NN import NN

# Создаем объект класса ReadData (читаем csv)
readData = ReadData('voice.csv')
# Создаем объект класса Page (для отображения стариц)
pages = Page(readData)
# Создаем объект класса PrepareData (получаем x_train, x_test, y_train, y_test и X, y)
prepare = Prepare(readData)
# Создаем объект класса logReg (модель логистической регрессии)
logReg = LogRegression(readData=readData, prepare=prepare, model=linear_model.LogisticRegression())
# Создаем объект класса NN (для отображения создания модели)
nn = NN(prepare)

# загружаем модель логистической регрессии
log_reg, lst_accu_stratified, x_train, x_test = logReg.log_reg_with_repeat_fold()
# переменная model
model = nn.create_model()
# получаем x_train, x_test, y_train, y_test и X, y
x_train, x_test, y_train, y_test = prepare.get_train_test_for_network()


# класс callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.94):
            print("\nReached 94% accuracy so cancelling training!")
            self.model.stop_training = True


# переменная callbacks
callbacks = MyCallback()

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



# выбор страницы
page = st.sidebar.selectbox("Выберите страницу", ["page 1", "page 2", "page 3",
                                                  "page 4", "page 5", "page 6",
                                                  "page 7", "page 8", "page 9",
                                                  "page 10"])

page_list = ["page 1", "page 2", "page 3",
             "page 4", "page 5", "page 6",
             "page 7", "page 8", "page 9",
             "page 10", "page 11", "page 12",
             "page 13", "page 14", "page 15", "model_testing"]

# отображение выбранной страницы
if page == "page 1":
    pages.page1()
elif page == "page 2":
    pages.page2()
elif page == "page 3":
    pages.page3()
elif page == "page 4":
    pages.page4()
elif page == "page 5":
    pages.page5()
elif page == "page 6":
    pages.page6()
elif page == "page 7":
    pages.page7()
elif page == "page 8":
    pages.page8()
elif page == "page 9":
    pages.page9()
elif page == "page 10":
    pages.page10()
elif page == "page 11":
    pages.page11()
elif page == "page 12":
    pages.page12()
elif page == "page 13":
    pages.page13()
elif page == "page 14":
    pages.page14()
elif page == "page 15":
    pages.page15()
