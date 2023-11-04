import joblib
from sklearn import linear_model
from sklearn.model_selection import RepeatedStratifiedKFold


class LogRegression:
    def __init__(self, readData, model):
        self.readData = readData
        self.model = model

        # модель логистической регрессии

        # функция для модели логистической регрессии
    def log_reg_with_repeat_fold(self):
        X = self.readData.get_dataframe().drop('label', axis=1)
        y = self.readData.get_dataframe()[['label']]
        rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
        lst_accu_stratified = []
        for train_index, test_index in rskf.split(X, y):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf = self.model.fit(x_train, y_train)
            lst_accu_stratified.append(self.model.score(x_test, y_test))
        # сохраним модель
        filename = 'log_reg.pkl'
        joblib.dump(clf, filename)
        return self.model, lst_accu_stratified, x_train, x_test

