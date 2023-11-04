# импорт необходимых библиотек
import streamlit as st
import numpy as np

from sklearn import linear_model

from model.ReadData import ReadData
from model.Page import Page
from model.LogRegression import LogRegression

readData = ReadData('voice.csv')
pages = Page(readData)
logReg = LogRegression(readData, model=linear_model.LogisticRegression())

# загружаем модель логистической регрессии
log_reg, lst_accu_stratified, x_train, x_test = logReg.log_reg_with_repeat_fold()


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
    page11()
elif page == "page 12":
    page12()
elif page == "page 13":
    page13()
elif page == "page 14":
    page14()
elif page == "page 15":
    page15()
