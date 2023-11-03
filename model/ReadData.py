import warnings

import pandas as pd


class ReadData:
    def __init__(self, file):
        self.file = file

        warnings.filterwarnings('ignore')
        pd.set_option('display.max_columns', None)
        # чтение датасета
        self.dataframe = pd.read_csv(file)
        # приведем названия колонок к нижнему регистру и удалим знаки препинания
        self.dataframe.columns = self.dataframe.columns.str.lower().str.replace('[^\w\s]', '', regex=True)
        # преобразуем строковые метки в числовые значения для колонки 'label': male = 1, female = 0
        dict = {'label': {'male': 1, 'female': 0}}
        self.dataframe.replace(dict, inplace=True)
        self.x = self.dataframe.loc[:, self.dataframe.columns != 'label']
        self.y = self.dataframe.loc[:, 'label']

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_dataframe(self):
        return self.dataframe
