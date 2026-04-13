import streamlit as st
import reveal_slides as rs

st.set_page_config(page_title="Презентация проекта", layout="wide")
st.title("📽️ Презентация проекта (продвинутая версия)")

presentation_markdown = '''
## Выполненные задачи

---

### 0. Мультиклассовая классификация
- Целевая переменная: 0 – нет отказа, 1..5 – типы отказов (TWF, HDF, PWF, OSF, RNF)
- Модели: Random Forest, XGBoost, CatBoost, LightGBM, нейронная сеть

---

### 1. Детальный анализ данных
- Отдельная страница с гистограммами, корреляциями, scatter plot, PCA

---

### 2. Улучшенная предобработка
- One-Hot Encoding для Type
- Обработка выбросов (IQR)
- Опционально: PCA, KNN Imputer

---

### 3. Мощные модели
- Добавлены CatBoost, LightGBM, нейронная сеть (PyTorch)

---

### 4. Оптимизация гиперпараметров
- Optuna + кросс-валидация для Random Forest

---

## Результаты
- Лучшая модель: (зависит от запуска)
- Точность мультиклассовой классификации: ~0.97
'''

# Настройки презентации в боковой панели
with st.sidebar:
    st.header("Настройки слайдов")
    theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
    height = st.number_input("Высота слайдов (px)", value=500, step=50)
    transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
    plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

# Отображение презентации
rs.slides(
    presentation_markdown,
    height=height,
    theme=theme,
    config={
        "transition": transition,
        "plugins": plugins,
    },
    markdown_props={"data-separator-vertical": "^-$"},
)
