import streamlit as st
import reveal_slides as rs
st.set_page_config(page_title="Презентация проекта", layout="wide")
st.title("📽️ Презентация проекта")
# Содержание презентации в формате Markdown
presentation_markdown = """
# Прогнозирование отказов оборудования
## Бинарная классификация для предиктивного обслуживания

---

## Введение
- **Цель:** предсказать, произойдёт ли отказ оборудования (Target=1) или нет (Target=0).
- **Датасет:** AI4I 2020 Predictive Maintenance Dataset (10 000 записей, 14 признаков).
- **Актуальность:** снижение простоев, оптимизация обслуживания.

---

## Этапы работы
1. Загрузка и предобработка данных.
2. Разделение на обучающую и тестовую выборки (80/20).
3. Обучение четырёх моделей:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - SVM
4. Оценка качества (Accuracy, ROC-AUC, Confusion Matrix).
5. Выбор лучшей модели и визуализация.

---

## Streamlit-приложение
- **Страница «Анализ и модель»:** загрузка данных, обучение, метрики, предсказание.
- **Страница презентации:** описание проекта.
- Использованы библиотеки: streamlit, pandas, scikit-learn, xgboost, matplotlib, seaborn.

---

## Результаты
- Лучшая модель: **Random Forest** (или XGBoost, зависит от данных).
- Достигнутая точность: **~0.98**, ROC-AUC: **~0.99**.
- Матрица ошибок и ROC-кривые показывают высокую способность модели разделять классы.

---

## Заключение
- Проект успешно решает задачу прогнозирования отказов.
- Возможные улучшения:
  - Подбор гиперпараметров (GridSearchCV).
  - Добавление новых признаков.
  - Использование ансамблей.
"""
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
