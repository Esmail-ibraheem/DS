import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# ------------------------------------
# Logging Configuration
# ------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------
# Title and Description
# ------------------------------------
st.title("💻👩🏻‍💻 NVIDIA AMD Intel MSI share prices")
st.header("Preprocessing and Visualization Application 🧹📊")

# ------------------------------------
# Data Engineering / Preprocessing Functions
# ------------------------------------
def load_data(files):
    """
    Load and combine multiple CSV/Excel files into one DataFrame.
    """
    dataframes = []
    for file in files:
        try:
            if file.name.endswith('.csv'):
                temp_df = pd.read_csv(file)
            else:
                temp_df = pd.read_excel(file)
            dataframes.append(temp_df)
        except Exception as e:
            st.error(f"Failed to load dataset from {file.name}: {e}")
            logging.error(f"Failed to load file {file.name}: {e}")
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"Combined {len(files)} files into one DataFrame with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
        return combined_df
    return None

def remove_outliers_iqr(dataframe):
    """
    Remove outliers using the IQR method.
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found. IQR outlier removal skipped.")
        logging.warning("No numeric columns found for IQR outlier removal.")
        return dataframe
    Q1 = dataframe[numeric_cols].quantile(0.25)
    Q3 = dataframe[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = dataframe[~((dataframe[numeric_cols] < (Q1 - 1.5 * IQR)) |
                              (dataframe[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    logging.info(f"Outlier removal (IQR) reduced rows from {len(dataframe)} to {len(filtered_df)}.")
    return filtered_df

def encode_categorical(dataframe):
    """
    Encode categorical variables using LabelEncoder.
    """
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        for column in categorical_columns:
            le = LabelEncoder()
            dataframe[column] = le.fit_transform(dataframe[column].astype(str))
        logging.info(f"Encoded {len(categorical_columns)} categorical columns.")
    return dataframe

def scale_features(dataframe):
    """
    Scale numeric features using StandardScaler.
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found. Feature scaling skipped.")
        logging.warning("No numeric columns found for scaling.")
        return dataframe
    scaler = StandardScaler()
    dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
    logging.info("Scaled numeric features using StandardScaler.")
    return dataframe

def handle_missing_values(dataframe):
    """
    Handle missing values by dropping rows with NA.
    """
    initial_rows = len(dataframe)
    dataframe = dataframe.dropna()
    logging.info(f"Dropped {initial_rows - len(dataframe)} rows due to missing values.")
    return dataframe

def preprocess_data(dataframe, steps):
    """
    Run selected preprocessing steps in order.
    """
    if "Handle Missing Values" in steps:
        dataframe = handle_missing_values(dataframe)
        st.info("Missing values handled by dropping rows with NA.")

    if "Encode Categorical Variables" in steps:
        dataframe = encode_categorical(dataframe)
        st.info("Categorical variables encoded.")

    if "Feature Scaling" in steps:
        dataframe = scale_features(dataframe)
        st.info("Features scaled.")

    if "Remove Outliers (IQR)" in steps:
        before_rows = len(dataframe)
        dataframe = remove_outliers_iqr(dataframe)
        st.info(f"IQR outlier removal done. {before_rows - len(dataframe)} rows removed.")

    return dataframe

# ------------------------------------
# Visualization Functions
# ------------------------------------
def get_visualization_options(df):
    """
    Return a list of available visualization options depending on the data.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Basic set of options
    options = []

    if len(numeric_cols) > 0:
        options.append("Histogram (first numeric col)")
        options.append("Box Plot (first numeric col)")
        options.append("Violin Plot (first numeric col)")
    if len(numeric_cols) > 1:
        options.append("Pairplot (numeric cols)")
        options.append("Correlation Heatmap (numeric cols)")
        options.append("Plotly Heatmap (numeric cols)")
        options.append("Scatter Plot (first two numeric cols)")
    if len(numeric_cols) >= 3:
        options.append("3D Scatter Plot (first three numeric cols)")

    if len(categorical_cols) > 0:
        options.append("Bar Chart (first categorical col)")
        options.append("Pie Chart (first categorical col)")

    if 'date' in df.columns:
        options.append("Line Chart by Date")

    return options

def create_single_visualization(data, choice):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Handle date conversions if date column is present
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
        except:
            pass

    # Create the selected visualization
    if choice == "Histogram (first numeric col)" and len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Box Plot (first numeric col)" and len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        ax.set_title(f'Box Plot of {col}')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Violin Plot (first numeric col)" and len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        sns.violinplot(x=data[col], ax=ax)
        ax.set_title(f'Violin Plot of {col}')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Pairplot (numeric cols)" and len(numeric_cols) > 1:
        pairplot_fig = sns.pairplot(data[numeric_cols])
        st.pyplot(pairplot_fig.fig)
        plt.close(pairplot_fig.fig)

    elif choice == "Correlation Heatmap (numeric cols)" and len(numeric_cols) > 1:
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Bar Chart (first categorical col)" and len(categorical_cols) > 0:
        col = categorical_cols[0]
        fig, ax = plt.subplots()
        data[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Chart of {col}')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Line Chart by Date" and 'date' in data.columns:
        data_with_date_index = data.set_index('date', inplace=False)
        fig, ax = plt.subplots()
        data_with_date_index.plot(ax=ax)
        ax.set_title('Line Chart of Date Series')
        st.pyplot(fig)
        plt.close(fig)

    elif choice == "Scatter Plot (first two numeric cols)" and len(numeric_cols) > 1:
        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title='Scatter Plot')
        st.plotly_chart(fig)

    elif choice == "Pie Chart (first categorical col)" and len(categorical_cols) > 0:
        fig = px.pie(data, names=categorical_cols[0], title='Pie Chart of ' + categorical_cols[0])
        st.plotly_chart(fig)

    elif choice == "Plotly Heatmap (numeric cols)" and len(numeric_cols) > 1:
        heatmap_data = data[numeric_cols].corr()
        fig = px.imshow(heatmap_data, text_auto=True, title='Heatmap of Numeric Variables')
        st.plotly_chart(fig)

    elif choice == "3D Scatter Plot (first three numeric cols)" and len(numeric_cols) >= 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=data[numeric_cols[0]],
            y=data[numeric_cols[1]],
            z=data[numeric_cols[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=data[numeric_cols[0]],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        fig.update_layout(title='3D Scatter Plot')
        st.plotly_chart(fig)

# ------------------------------------
# Sidebar - File Upload
# ------------------------------------
st.sidebar.header("Upload Multiple CSV or Excel Files")
uploaded_files = st.sidebar.file_uploader("Upload your files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

df = None

if uploaded_files:
    df = load_data(uploaded_files)
    if df is not None:
        st.success("All datasets combined successfully!")
        st.subheader("Preview of Combined Dataset")
        st.dataframe(df.head())

# ------------------------------------
# Sidebar - Data Cleaning Code
# ------------------------------------
with st.sidebar.expander("🔧 Show Data Cleaning Code"):
    code_snippet = r"""
def preprocess_data(dataframe, steps):
    if "Handle Missing Values" in steps:
        dataframe = handle_missing_values(dataframe)
    if "Encode Categorical Variables" in steps:
        dataframe = encode_categorical(dataframe)
    if "Feature Scaling" in steps:
        dataframe = scale_features(dataframe)
    if "Remove Outliers (IQR)" in steps:
        dataframe = remove_outliers_iqr(dataframe)
    return dataframe
    """
    st.code(code_snippet, language='python')

# Always show a preview of the current dataset in the sidebar if df is not None
if df is not None:
    st.sidebar.subheader("Preview of Current Dataset")
    st.sidebar.dataframe(df.head())

# ------------------------------------
# Preprocessing Steps
# ------------------------------------
if df is not None:
    st.subheader("Data Preprocessing")

    if st.checkbox("Show dataset info"):
        buffer = []
        df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)

    if st.checkbox("Show missing values summary"):
        st.write(df.isnull().sum())

    preprocess_options = st.multiselect(
        "Select preprocessing steps (applied in order):",
        [
            "Handle Missing Values",
            "Encode Categorical Variables",
            "Feature Scaling",
            "Remove Outliers (IQR)"
        ]
    )

    if st.button("Preprocess Data 🧹"):
        try:
            df = preprocess_data(df, preprocess_options)
            st.success("Data preprocessing completed!")
            st.dataframe(df.head())

            # Update sidebar preview after preprocessing
            st.sidebar.subheader("Preview of Preprocessed Data")
            st.sidebar.dataframe(df.head())
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            logging.error(f"Preprocessing error: {e}")

    # ------------------------------------
    # Visualization Selection
    # ------------------------------------
    st.subheader("Data Visualization")
    st.write("Choose a visualization to run from the dropdown. This helps avoid running all visualizations at once.")

    vis_options = get_visualization_options(df)
    if len(vis_options) > 0:
        selected_vis = st.selectbox("Select a visualization:", vis_options)

        if st.button("Run Selected Visualization"):
            create_single_visualization(df, selected_vis)
    else:
        st.write("No visualizations available. Check if data has numeric or categorical columns.")

    # ------------------------------------
    # Download Processed Data
    # ------------------------------------
    if st.button("Download Processed Data"):
        try:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error preparing data for download: {e}")
            logging.error(f"Error preparing download: {e}")
