from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf


class NN:
    def __init__(self, prepare):
        self.prepare = prepare
        X, y = self.prepare.get_X_y()
        self.input_shape = (X.shape[1],)

        # нейронная сеть

    # определим функцию create_model, которая создает нейронную сеть
    def create_model(self):
        # переменная input_shape
        model = Sequential()
        # входной слой
        model.add(Dense(256, input_shape=self.input_shape, activation="relu"))
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
