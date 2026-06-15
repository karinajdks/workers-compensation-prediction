import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def analysis_and_model_page():
    st.title("📈 Прогнозирование стоимости страховых выплат")

    if st.button("📂 Загрузить данные с OpenML"):
        with st.spinner("Загрузка данных... (100,000 записей)"):
            data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
            df = data.frame
            st.session_state['df'] = df
        st.success("✅ Данные успешно загружены!")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        st.subheader("🔍 Первые строки данных")
        st.dataframe(df.head())

        st.subheader("📊 Статистика данных")
        st.write(df.describe())

        # Предобработка
        st.subheader("🛠️ Предобработка данных")

        # Обработка datetime
        df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
        df['DateReported'] = pd.to_datetime(df['DateReported'])
        df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
        df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
        df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
        df = df.drop(columns=['DateTimeOfAccident', 'DateReported'])

        # Преобразование категориальных переменных
        cat_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            st.session_state[f'encoder_{col}'] = le

        # Логарифмическое преобразование целевой переменной
        y = np.log1p(df['UltimateIncurredClaimCost'])
        X = df.drop(columns=['UltimateIncurredClaimCost'])
        
        # Масштабирование
        num_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                        'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                        'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']
        scaler = StandardScaler()
        X[num_features] = scaler.fit_transform(X[num_features])
        st.session_state['scaler'] = scaler
        st.session_state['num_features'] = num_features

        st.write(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        st.success("✅ Предобработка выполнена!")

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("🤖 Обучение моделей")

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        results = []
        for name, model in models.items():
            with st.spinner(f"Обучение {name}..."):
                model.fit(X_train, y_train)
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_test)
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                results.append({
                    "Модель": name,
                    "MAE": f"${mae:,.0f}",
                    "RMSE": f"${rmse:,.0f}",
                    "R²": f"{r2:.4f}"
                })
                st.session_state[f"model_{name}"] = model

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        best_r2 = max([float(r["R²"]) for r in results])
        best_model_name = [r["Модель"] for r in results if float(r["R²"]) == best_r2][0]
        st.success(f"🏆 Лучшая модель: **{best_model_name}** с R² = {best_r2:.4f}")

        # Анализ: почему R² низкий
        st.info("""
        **📌 Анализ результатов:**
        
        Полученный R² ≈ 0.09 указывает на то, что предоставленные признаки слабо коррелируют с итоговой стоимостью выплат.
        
        **Возможные причины:**
        - В датасете отсутствуют ключевые факторы (тяжесть травмы, качество лечения)
        - Большое количество выбросов и шума в данных
        - Сложность прогнозирования страховых выплат в реальности
        
        **Предложения по улучшению:**
        - Добавить медицинские признаки (диагноз, лечение)
        - Учитывать судебные издержки
        - Использовать историю предыдущих выплат по аналогичным случаям
        """)

        # Важность признаков
        st.subheader("📌 Важность признаков")
        rf_model = models["Random Forest"]
        importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': rf_model.feature_importances_
        }).sort_values('Важность', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Признак'][:10], importance['Важность'][:10])
        ax.set_xlabel("Важность")
        ax.set_title("Топ-10 наиболее важных признаков")
        ax.invert_yaxis()
        st.pyplot(fig)

        # График предсказаний
        st.subheader("📈 Сравнение предсказаний с реальными значениями")
        best_model = st.session_state[f"model_{best_model_name}"]
        y_pred_log_best = best_model.predict(X_test)
        y_pred_best = np.expm1(y_pred_log_best)
        y_true_best = np.expm1(y_test)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sample_idx = np.random.choice(len(y_true_best), size=min(2000, len(y_true_best)), replace=False)
        ax2.scatter(y_true_best.iloc[sample_idx], y_pred_best[sample_idx], alpha=0.3, s=10)
        ax2.plot([y_true_best.min(), y_true_best.max()], [y_true_best.min(), y_true_best.max()], 'r--', lw=2)
        ax2.set_xlabel("Реальные значения ($)")
        ax2.set_ylabel("Предсказанные значения ($)")
        ax2.set_title(f"{best_model_name}: Предсказания vs Реальность\nR² = {best_r2:.4f}")
        st.pyplot(fig2)

        # Предсказание
        st.header("🔮 Предсказание нового страхового случая")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("📅 Возраст", 13, 76, 34)
                gender = st.selectbox("👤 Пол", ["M", "F"])
                weekly_pay = st.number_input("💰 Зарплата ($)", 0, 5000, 750)
                dependent_children = st.number_input("👶 Дети на иждивении", 0, 10, 0)

            with col2:
                hours_worked = st.number_input("⏱️ Часов в неделю", 0, 80, 40)
                days_worked = st.number_input("📆 Дней в неделю", 0, 7, 5)
                initial_estimate = st.number_input("💵 Начальная оценка ($)", 0, 100000, 5000)
                marital_status = st.selectbox("💍 Семейное положение", ["Single", "Married", "Divorced", "Widowed"])

            submitted = st.form_submit_button("🎯 Предсказать")

            if submitted:
                from datetime import datetime
                accident_month = datetime.now().month
                accident_dayofweek = datetime.now().weekday()
                reporting_delay = 7

                gender_enc = 1 if gender == "M" else 0
                marital_enc = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}[marital_status]
                part_time_enc = 0
                claim_enc = 0

                input_data = np.array([[
                    age, dependent_children, 0, weekly_pay,
                    hours_worked, days_worked, initial_estimate, accident_month,
                    accident_dayofweek, reporting_delay, gender_enc,
                    marital_enc, part_time_enc, claim_enc
                ]])

                scaler = st.session_state['scaler']
                input_data[:, :10] = scaler.transform(input_data[:, :10])

                best_model = st.session_state[f"model_{best_model_name}"]
                pred_log = best_model.predict(input_data)[0]
                prediction = np.expm1(pred_log)

                st.success(f"### Прогнозируемая стоимость: **${prediction:,.2f}**")

if __name__ == "__main__":
    analysis_and_model_page()
