import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Lasso,
    Ridge,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

def run_comparative_analysis():
    st.title("Comparative Analysis of Machine Learning and Data mining Algorithms 🔍📋")

    # Dataset Upload
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

        # Display Dataset
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Data Preprocessing
        st.header("Data Preprocessing")
        st.write("Automatically handling missing values, encoding categorical variables, and scaling features.")

        # Handle missing values
        df = df.dropna()

        # Encode categorical variables
        label_encoders = {}
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Feature Scaling
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        st.success("Data preprocessing completed!")

        # Target Variable Selection
        st.header("Select Target Variable")
        all_columns = df.columns.tolist()
        target_variable = st.selectbox("Select the target variable:", all_columns)
        feature_variables = [col for col in all_columns if col != target_variable]

        # Determine Task Type
        st.header("Determining Task Type")
        y = df[target_variable]
        if y.dtype == 'object' or y.dtype.name == 'category':
            task_type = 'Classification'
        elif np.issubdtype(y.dtype, np.number):
            unique_values = y.nunique()
            if unique_values <= 20:
                task_type = 'Classification'
                st.info("Target variable has a small number of unique numeric values; treating as classification.")
                y = y.astype(int)
            else:
                task_type = 'Regression'
        else:
            st.error("Unknown target variable type.")
            return

        st.write(f"Detected task type: **{task_type}**")

        # Split the data
        X = df[feature_variables]
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Encode target variable if classification
        if task_type == 'Classification':
            if y_train.dtype == 'object' or y_train.dtype.name == 'category':
                target_le = LabelEncoder()
                y_train = target_le.fit_transform(y_train)
                y_test = target_le.transform(y_test)
        else:
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

        # Define Algorithms
        if task_type == 'Classification':
            models = {
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Support Vector Machine': SVC(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Neural Network': MLPClassifier(max_iter=1000)
            }
        else:
            models = {
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Linear Regression': LinearRegression(),
                'Lasso Regression': Lasso(),
                'Ridge Regression': Ridge(),
                'Support Vector Machine': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Neural Network': MLPRegressor(max_iter=1000)
            }

        # Initialize results storage
        results = {}
        st.header("Model Training and Evaluation")
        for name, model in models.items():
            st.subheader(f"Algorithm: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == 'Classification':
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.4f}")
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.title(f'Confusion Matrix - {name}')
                st.pyplot(fig)
                results[name] = acc
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"R-squared: {r2:.4f}")
                # Scatter Plot
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                plt.title(f'Actual vs Predicted - {name}')
                st.pyplot(fig)
                results[name] = r2

            # Feature Importance or Coefficients
            if hasattr(model, 'feature_importances_'):
                st.write("Feature Importances:")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(importance_df.set_index('Feature'))

            elif hasattr(model, 'coef_'):
                st.write("Feature Coefficients:")
                coefficients = model.coef_
                coef_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Coefficient': coefficients.flatten()
                }).sort_values(by='Coefficient', ascending=False)
                st.bar_chart(coef_df.set_index('Feature'))

            # Plot Decision Tree
            if name == 'Decision Tree':
                st.write("Decision Tree Visualization:")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, feature_names=X_train.columns, filled=True, ax=ax)
                st.pyplot(fig)

            # Plot Random Forest Trees (First Tree)
            if name == 'Random Forest':
                st.write("Random Forest - First Tree Visualization:")
                fig, ax = plt.subplots(figsize=(20, 10))
                estimator = model.estimators_[0]
                plot_tree(estimator, feature_names=X_train.columns, filled=True, ax=ax)
                st.pyplot(fig)

            st.write("---")

        # Compare Models
        st.header("Model Comparison")
        if task_type == 'Classification':
            result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
        else:
            result_df = pd.DataFrame.from_dict(results, orient='index', columns=['R-squared'])
        st.table(result_df)

        st.bar_chart(result_df)

        # Data Visualization
        st.header("Data Visualization")
        if st.checkbox("Show Pairplot"):
            try:
                fig = sns.pairplot(df)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating pairplot: {e}")

        if st.checkbox("Show Correlation Heatmap"):
            corr = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Download Processed Data
        if st.button("Download Processed Data"):
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    run_comparative_analysis()
