import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb

st.set_page_config(page_title="Анализ и модель", layout="wide")
st.title("📊 Анализ данных и обучение модели")

# Инициализация состояния сессии
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Функция предобработки данных 
def preprocess_data(df):
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df.columns = (df.columns
                  .str.replace('[', '', regex=False)
                  .str.replace(']', '', regex=False)
                  .str.replace(' ', '_'))
    # Кодирование Type
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    # Целевая переменная и признаки
    X = df.drop('Machine_failure', axis=1)
    y = df['Machine_failure']
    # Масштабирование числовых признаков 
    num_features = ['Air_temperature_K', 'Process_temperature_K',
                    'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    return X, y, le, scaler
# Функция обучения моделей
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                                     use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    results = {}
    best_auc = -1
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        results[name] = {
            'model': model,
            'accuracy': acc,
            'roc_auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model
    return results, best_model_name, best_model_obj
# Загрузка данных
uploaded_file = st.file_uploader("📂 Загрузите CSV-файл с данными", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")
    with st.expander("Показать первые строки данных"):
        st.dataframe(df.head())
else:
    import os
    default_path = "data/predictive_maintenance.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"Используются данные из файла {default_path}")
        with st.expander("Показать первые строки данных"):
            st.dataframe(df.head())
    else:
        st.warning("Пожалуйста, загрузите CSV-файл.")
        st.stop()
# Кнопка обучения моделей
if st.button("🚀 Обучить модели", type="primary"):
    with st.spinner("Идёт предобработка и обучение..."):
        X, y, le, scaler = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Сохраняем объекты для предсказания
        st.session_state.scaler = scaler
        st.session_state.label_encoder = le
        st.session_state.feature_names = X.columns.tolist()
        # Обучение
        results, best_name, best_model = train_models(X_train, y_train, X_test, y_test)
        st.session_state.best_model = best_model
        st.session_state.results = results
        st.session_state.best_name = best_name
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    st.success(f"✅ Обучение завершено! Лучшая модель: **{best_name}**")
    # Отображение метрик
    st.subheader("📈 Сравнение моделей")
    metrics_df = pd.DataFrame({
        'Модель': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'ROC-AUC': [results[m]['roc_auc'] for m in results]
    }).round(4)
    st.dataframe(metrics_df)
    # Лучшая модель: отчёт
    st.subheader(f"🏆 Лучшая модель: {best_name}")
    best_res = results[best_name]
    st.write(f"**Accuracy:** {best_res['accuracy']:.4f}")
    st.write(f"**ROC-AUC:** {best_res['roc_auc']:.4f}")
    # Матрица ошибок
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, best_res['y_pred'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истина')
    st.pyplot(fig)
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, best_res['y_pred'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))
    # ROC-кривые всех моделей
    st.subheader("ROC-кривые")
    fig2, ax2 = plt.subplots()
    for name, res in results.items():
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax2.plot([0, 1], [0, 1], 'k--', label='Случайный')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    st.pyplot(fig2)
# Блок предсказания для новых данных
st.header("🔮 Предсказание по новым данным")
if st.session_state.best_model is not None:
    st.write("Введите значения признаков (в оригинальных единицах):")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            product_type = st.selectbox("Тип продукта (Type)", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
            process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
        with col2:
            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500, step=10)
            torque = st.number_input("Torque [Nm]", value=40.0, step=0.1)
            tool_wear = st.number_input("Tool wear [min]", value=100, step=1)
        submitted = st.form_submit_button("📌 Предсказать")
        if submitted:
            # Используем те же имена, что и после переименования
            input_dict = {
                'Type': [product_type],
                'Air_temperature_K': [air_temp],
                'Process_temperature_K': [process_temp],
                'Rotational_speed_rpm': [rotational_speed],
                'Torque_Nm': [torque],
                'Tool_wear_min': [tool_wear]
            }
            input_df = pd.DataFrame(input_dict)
            # Кодируем Type
            input_df['Type'] = st.session_state.label_encoder.transform(input_df['Type'])
            # Масштабируем числовые признаки
            num_features = ['Air_temperature_K', 'Process_temperature_K',
                            'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
            input_df[num_features] = st.session_state.scaler.transform(input_df[num_features])
            # Предсказание
            pred = st.session_state.best_model.predict(input_df)[0]
            proba = st.session_state.best_model.predict_proba(input_df)[0][1]
            st.write("### Результат:")
            if pred == 1:
                st.error(f"⚠️ Отказ оборудования **предсказан** (вероятность отказа: {proba:.2f})")
            else:
                st.success(f"✅ Отказ оборудования **не предсказан** (вероятность отказа: {proba:.2f})")
else:
    st.info("Сначала обучите модель, нажав кнопку выше.")
