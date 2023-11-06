from sklearn.model_selection import RepeatedStratifiedKFold


class Prepare:
    def __init__(self, readData):
        self.readData = readData

        # функция для подготовки x_train, x_test, y_train, y_test

    def get_train_test_for_network(self):
        X = self.readData.get_dataframe().drop('label', axis=1)
        y = self.readData.get_dataframe()[['label']]
        rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
        for train_index, test_index in rskf.split(X, y):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return x_train, x_test, y_train, y_test

    # функция для получения Х у
    def get_X_y(self):
        X = self.readData.get_dataframe().drop('label', axis=1)
        y = self.readData.get_dataframe()[['label']]
        return X, y
