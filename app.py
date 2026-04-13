import streamlit as st

pages = {
    "Анализ и модель": [st.Page("analysis_and_model.py", title="Анализ и модель")],
    "Детальный анализ данных": [st.Page("EDA_Exploratory_Data_Analysis.py", title="EDA")],
    "Презентация": [st.Page("presentation.py", title="Презентация")],
}

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
