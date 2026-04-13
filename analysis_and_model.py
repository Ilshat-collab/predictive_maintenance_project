import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler

st.set_page_config(page_title="Анализ и модель (продвинутая)", layout="wide")
st.title("🚀 Продвинутый анализ и обучение моделей")

# Функция загрузки и подготовки
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        import os
        if os.path.exists("data/predictive_maintenance.csv"):
            df = pd.read_csv("data/predictive_maintenance.csv")
        else:
            st.error("Файл данных не найден. Загрузите CSV-файл.")
            st.stop()
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    def get_failure_type(row):
        for ft in failure_types:
            if row[ft] == 1:
                return failure_types.index(ft) + 1
        return 0
    df['failure_class'] = df.apply(get_failure_type, axis=1)
    return df
# Загрузка данных через интерфейс
uploaded_file = st.file_uploader("📂 Загрузите CSV-файл с данными", type="csv")
if uploaded_file is not None:
    df = load_and_prepare_data(uploaded_file)
    st.success("Файл успешно загружен!")
else:
    import os
    if os.path.exists("data/predictive_maintenance.csv"):
        df = load_and_prepare_data()
        st.info("Используются данные из папки data/")
    else:
        st.warning("Пожалуйста, загрузите CSV-файл или поместите файл в data/predictive_maintenance.csv")
        st.stop()
st.write("Мультиклассовая цель: 0 – нет отказа, 1..5 – тип отказа")

st.header("Предобработка данных")
handle_outliers = st.checkbox("Обработать выбросы (IQR)", value=True)
use_pca = st.checkbox("Применить PCA (уменьшение размерности до 5 компонент)", value=False)
impute_missing = st.checkbox("Применить KNN Imputer (заполнение пропусков)", value=False)
# Определяем признаки
numeric_features = ['Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']
# Исходные данные
X_raw = df.drop(['failure_class', 'Machine failure', 'UDI', 'Product ID',
                 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['failure_class']
# Обработка выбросов
if handle_outliers:
    for col in numeric_features:
        Q1 = X_raw[col].quantile(0.25)
        Q3 = X_raw[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_raw[col] = X_raw[col].clip(lower, upper)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2,
                                                    random_state=42, stratify=y)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if impute_missing:
    imputer = KNNImputer(n_neighbors=5)
    X_train_processed = imputer.fit_transform(X_train_processed)
    X_test_processed = imputer.transform(X_test_processed)
# PCA
if use_pca:
    pca = PCA(n_components=5)
    X_train_processed = pca.fit_transform(X_train_processed)
    X_test_processed = pca.transform(X_test_processed)
    st.write(f"Размерность после PCA: {X_train_processed.shape[1]}")

st.write(f"Размер обучающей выборки: {X_train_processed.shape}")
st.write(f"Размер тестовой: {X_test_processed.shape}")
# 3. Обучение моделей 
st.header("Обучение моделей")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, objective='multi:softmax', num_class=6, random_state=42),
    "CatBoost": cb.CatBoostClassifier(iterations=100, verbose=0, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}
# Простая нейронная сеть (PyTorch)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_nn(X_tr, y_tr, X_te, y_te, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Преобразование в тензоры
    X_tr_t = torch.tensor(X_tr.astype(np.float32))
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.long)
    X_te_t = torch.tensor(X_te.astype(np.float32))
    y_te_t = torch.tensor(y_te.values, dtype=torch.long)
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    epochs = 30
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(X_te_t.to(device))
        _, pred = torch.max(outputs, 1)
        acc = (pred.cpu().numpy() == y_te).mean()
    return model, acc

if st.button("🚀 Обучить все модели"):
    results = {}
    # Обучение классических моделей
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'accuracy': acc, 'f1': f1, 'model': model, 'y_pred': y_pred}
        st.write(f"**{name}** - Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
    # Обучение нейронных сетей
    st.write("**Нейронная сеть (PyTorch)**")
    nn_model, nn_acc = train_nn(X_train_processed, y_train, X_test_processed, y_test, X_train_processed.shape[1])
    results['Neural Network'] = {'accuracy': nn_acc, 'f1': 0, 'model': nn_model}
    st.write(f"Accuracy: {nn_acc:.4f}")
    # Выбор лучшей модели по accuracy
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['accuracy']
    st.success(f"Лучшая модель: **{best_name}** с Accuracy {best_acc:.4f}")
    # Матрица ошибок для лучшей модели
    if best_name != 'Neural Network':
        y_pred_best = results[best_name]['y_pred']
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test_t = torch.tensor(X_test_processed.astype(np.float32)).to(device)
        best_model.eval()
        with torch.no_grad():
            outputs = best_model(X_test_t)
            _, y_pred_best = torch.max(outputs, 1)
            y_pred_best = y_pred_best.cpu().numpy()
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    st.pyplot(fig)
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred_best, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))
# 4. Оптимизация гиперпараметров с Optuna 
st.header("Оптимизация гиперпараметров (Optuna)")
if st.button("🔧 Запустить оптимизацию для Random Forest"):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_processed, y_train, cv=skf, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    st.write("Лучшие гиперпараметры:", study.best_params)
    st.write("Лучшая accuracy (CV):", study.best_value)

    best_rf = RandomForestClassifier(**study.best_params, random_state=42)
    best_rf.fit(X_train_processed, y_train)
    y_pred_opt = best_rf.predict(X_test_processed)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    st.write(f"Точность на тесте после оптимизации: {acc_opt:.4f}")
