import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go  # Import this module

def run2DVisualization():
    # Function to create various visualizations from the data
    def create_visualizations(data):
        plots = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # Histograms for numeric columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_title(f'Histogram of {col}')
            plots.append(fig)

        # Box plots for numeric columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=data[col], ax=ax)
            ax.set_title(f'Box Plot of {col}')
            plots.append(fig)

        # Scatter plot matrix for numeric columns
        if len(numeric_cols) > 1:
            sns_pairplot = sns.pairplot(data[numeric_cols])
            plots.append(sns_pairplot.fig)

        # Correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            corr = data[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            plots.append(fig)

        # Bar charts for categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                fig, ax = plt.subplots()
                data[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Chart of {col}')
                plots.append(fig)

        # Line charts (if a 'date' column is present)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data_with_date_index = data.set_index('date', inplace=False)
            fig, ax = plt.subplots()
            data_with_date_index.plot(ax=ax)
            ax.set_title('Line Chart of Date Series')
            plots.append(fig)

        # Scatter plot using Plotly
        if len(numeric_cols) >= 2:
            fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title='Scatter Plot')
            plots.append(fig)

        # Pie chart for categorical columns (only the first categorical column)
        if len(categorical_cols) > 0:
            fig = px.pie(data, names=categorical_cols[0], title='Pie Chart of ' + categorical_cols[0])
            plots.append(fig)

        # Heatmap for numeric columns
        if len(numeric_cols) > 1:
            heatmap_data = data[numeric_cols].corr()
            fig = px.imshow(heatmap_data, text_auto=True, title='Heatmap of Numeric Variables')
            plots.append(fig)

        # Violin plots for numeric columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.violinplot(x=data[col], ax=ax)
            ax.set_title(f'Violin Plot of {col}')
            plots.append(fig)

        return plots

    # Function to analyze the uploaded data
    def analyze_data(file_input):
        try:
            # Read the CSV file
            data = pd.read_csv(file_input)
        except UnicodeDecodeError:
            # If utf-8 fails, try reading it with 'ISO-8859-1' encoding
            file_input.seek(0)  # Reset the pointer to the beginning
            data = pd.read_csv(file_input, encoding='ISO-8859-1')
        # Generate visualizations
        visualizations = create_visualizations(data)
        return data, visualizations

    # Streamlit interface
    st.title('DATA BOARD 📊')
    st.write('Upload a `.csv` file to generate various visualizations and interactive plots.')

    # File upload component
    file_input = st.file_uploader('Upload your `.csv` file', type=['csv'])

    # Analyze data and generate visualizations when a file is uploaded
    if file_input is not None:
        data, visualizations = analyze_data(file_input)

        # Display data table
        st.write('## Data Table')
        st.dataframe(data)

        # Display visualizations
        st.write('## Visualizations')

        for viz in visualizations:
            if isinstance(viz, plt.Figure):
                st.pyplot(viz)
            elif isinstance(viz, go.Figure):  # Corrected here
                st.plotly_chart(viz)
            else:
                # For seaborn pairplot, which returns a PairGrid object
                try:
                    st.pyplot(viz)
                except:
                    pass
