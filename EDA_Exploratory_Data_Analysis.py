import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

st.set_page_config(page_title="Детальный анализ данных", layout="wide")
st.title("📊 Детальный анализ датасета")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Детальный анализ данных", layout="wide")
st.title("📊 Детальный анализ датасета")

# Загрузка данных
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        import os
        if os.path.exists("data/predictive_maintenance.csv"):
            df = pd.read_csv("data/predictive_maintenance.csv")
        else:
            st.error("Файл данных не найден. Загрузите CSV-файл.")
            st.stop()
    return df
uploaded_file = st.file_uploader("📂 Загрузите CSV-файл для анализа", type="csv", key="eda_uploader")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("Файл загружен!")
else:
    import os
    if os.path.exists("data/predictive_maintenance.csv"):
        df = load_data()
        st.info("Используются данные из папки data/")
    else:
        st.warning("Пожалуйста, загрузите CSV-файл или поместите его в data/predictive_maintenance.csv")
        st.stop()

st.header("1. Общая информация")
st.write(f"Количество записей: {df.shape[0]}, признаки: {df.shape[1]}")
st.dataframe(df.head())
st.header("2. Распределение числовых признаков")
num_cols = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col)
axes[-1].axis('off')
st.pyplot(fig)
st.header("3. Корреляционная матрица")
corr = df[num_cols].corr()
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)
st.header("4. Scatter plot: взаимосвязи признаков")
x_axis = st.selectbox("Ось X", num_cols)
y_axis = st.selectbox("Ось Y", num_cols)
fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color='Machine failure',
                         title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig_scatter)

st.header("5. Анализ целевой переменной (типы отказов)")
failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_counts = df[failure_cols].sum()
fig_fail = px.bar(x=failure_counts.index, y=failure_counts.values,
                  title="Количество отказов по типам")
st.plotly_chart(fig_fail)

st.subheader("Распределение отказов по признакам (Boxplot)")
for col in num_cols:
    fig_box = px.box(df, x='Machine failure', y=col,
                     title=f"{col} по наличию отказа")
    st.plotly_chart(fig_box)

st.header("6. PCA проекция (уменьшение размерности)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[num_cols])
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['failure'] = df['Machine failure'].astype(str)
fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='failure',
                     title="PCA визуализация")
st.plotly_chart(fig_pca)

st.header("7. Анализ выбросов (IQR)")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    st.write(f"**{col}**: выбросов {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
