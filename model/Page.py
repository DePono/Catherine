import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from model.ReadData import ReadData


class Page:
    def __init__(self, readData):
        self.readData = readData

    def page1(self):
        st.title('page1: heatmap')
        st.write('тепловая карта корреляций')
        plt.figure(figsize=(15, 10), dpi=100)
        sns.heatmap(self.readData.get_dataframe().corr(), cmap="viridis", annot=True, linewidth=0.5)
        st.pyplot(plt)

    # page 2
    def page2(self):
        st.title('page2: median graph')
        st.write('график медианы для диапазона доминирующей частоты в акустическом сигнале')
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=4.94,
            title={'text': 'медиана для диапазона доминирующей частоты в акустическом сигнале'},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        st.plotly_chart(fig)

    # page 3
    def page3(self):
        st.title('page3: median and maximum value graph')
        st.write(
            'график сравнения медианы и максимального значения для диапазона доминирующей частоты в акустическом сигнале')
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            gauge={'shape': "bullet"},
            delta={'reference': 21.5},
            value=4.94,
            domain={'x': [0.1, 1], 'y': [0.2, 0.9]}
        ))
        fig.update_layout(
            title_text='сравнение медианы и максимального значения для диапазона доминирующей частоты в акустическом сигнале',
            title_x=0.5)
        st.plotly_chart(fig)

    # page 4
    def page4(self):
        st.title('page4: расстояние от STD и MEAN для *Mean Dom*')
        st.write('график расстояния от STD (стандартного отклонения) и MEAN (среднего значения) для *Mean Dom*')
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            gauge={'shape': "bullet"},
            delta={'reference': 0.08},
            value=0.06,
            domain={'x': [0.1, 1], 'y': [0.2, 0.9]}
        ))
        fig.update_layout(title_text='расстояние от STD (стандартное отклонение) и MEAN (среднее значение) для *Mean '
                                     'Dom*', title_x=0.4)
        st.plotly_chart(fig)

    # page 5
    def page5(self):
        st.title('page5: расстояние от STD и MEAN для *IQR*')
        st.write('график расстояния от STD (стандартного отклонения) и MEAN (среднего значения) для *IQR*')
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            gauge={'shape': "bullet"},
            delta={'reference': 0.25},
            value=0.08,
            domain={'x': [0.1, 1], 'y': [0.2, 0.9]}
        ))
        fig.update_layout(title_text='расстояние от STD (стандартное отклонение) и MEAN (среднее значение) для *IQR*',
                          title_x=0.4)
        st.plotly_chart(fig)

    # page 6
    def page6(self):
        st.title('page6: распределение данных и их связь')
        st.write(
            'распределение данных и их связь между "q25" и "q75", а также их связь с категорией "label (male, female)"')
        fig = px.scatter(self.readData.get_dataframe(), x="q25", y="q75", color="label", size='median',
                         hover_data=['label'])
        st.plotly_chart(fig)

    # page 7
    def page7(self):
        st.title('page7: анализ корреляции с "label"')
        st.write('столбцы имеющие наибольшую корреляцию с колонкой "label"')

        def target_coeff(dataframe, target):
            data = dataframe.corr()[target].sort_values(ascending=False)
            indices = data.index
            labels = []
            corr = []
            for i in range(1, len(indices)):
                labels.append(indices[i])
                corr.append(data[i])
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            sns.barplot(x=corr, y=labels, palette="RdBu", ax=ax)
            plt.title(f'Correlation Coefficient for {target.upper()} column')
            st.pyplot(fig)

        target_coeff(self.readData.get_dataframe(), 'label')

    # page 8
    def page8(self):
        st.title('page8: средняя частота')
        st.write('распределение средней частоты для "meanfun".')
        fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
        sns.distplot(self.readData.get_dataframe()['meanfun'], kde=False, bins=30, ax=ax)
        values = np.array([rec.get_height() for rec in ax.patches])
        norm = plt.Normalize(values.min(), values.max())
        colors = plt.cm.jet(norm(values))
        for rec, col in zip(ax.patches, colors):
            rec.set_color(col)
        plt.title('Distribution of Mean Frequence', size=20, color='black')
        st.pyplot(fig)

    # page 9
    def page9(self):
        st.title('page9: диапазон доминирующей частоты')
        st.write('распределение диапазона доминирующей частоты в акустическом сигнале.')
        fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
        sns.distplot(self.readData.get_dataframe()['dfrange'], kde=False, bins=30, ax=ax)
        values = np.array([rec.get_height() for rec in ax.patches])
        norm = plt.Normalize(values.min(), values.max())
        colors = plt.cm.jet(norm(values))
        for rec, col in zip(ax.patches, colors):
            rec.set_color(col)
        plt.title('распределение диапазона доминирующей частоты в акустическом сигнале', size=20, color='black')
        st.pyplot(fig)

    # page 10
    def page10(self):
        st.title('page10: KDE')
        st.write('KDE-графики для каждого признака, разделенные по классам (female = 0, male = 1).')
        fig, axes = plt.subplots(4, 5, figsize=(20, 20))
        features = self.readData.get_dataframe().columns[:-1]  # Исключаем последний столбец 'label'
        for i in range(4):
            for j in range(5):
                k = i * 5 + j + 1
                ax = axes[i, j]
                if k <= 20:
                    ax.set_title(features[k - 1])
                    sns.kdeplot(
                        self.readData.get_dataframe().loc[self.readData.get_dataframe()['label'] == 0, features[k - 1]],
                        color='green', label='F', ax=ax)
                    sns.kdeplot(
                        self.readData.get_dataframe().loc[self.readData.get_dataframe()['label'] == 1, features[k - 1]],
                        color='red', label='M', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
