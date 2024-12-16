# DS

### Step-by-Step Guide to Clean a Local Dataset Using Python

1. **Import Necessary Libraries**

   ```python
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   ```

2. **Load the Local Dataset**

   Replace `'path_to_your_dataset.csv'` with the actual file path to your dataset.

   ```python
   # Load the dataset
   file_path = 'path_to_your_dataset.csv'
   df = pd.read_csv(file_path)
   ```

3. **Inspect the Dataset**

   ```python
   # Display the first few rows
   print(df.head())

   # Check the shape of the dataset
   print(df.shape)

   # Check data types and missing values
   print(df.info())

   # Check for missing values
   print(df.isnull().sum())

   # Check for duplicate rows
   print(df.duplicated().sum())
   ```

4. **Handle Missing Values**

   Decide whether to drop or fill missing values based on the context.

   ```python
   # Example: Drop rows where 'Fuel_Consumption' is missing
   df.dropna(subset=['Fuel_Consumption'], inplace=True)

   # Example: Fill missing values in 'CO2_Emissions' with the median
   df['CO2_Emissions'].fillna(df['CO2_Emissions'].median(), inplace=True)
   ```

5. **Handle Duplicates**

   Remove duplicate rows if necessary.

   ```python
   # Drop duplicate rows
   df.drop_duplicates(inplace=True)
   ```

6. **Clean Specific Columns**

   - **Text Columns**

     ```python
     # Example: Clean 'Ship_Name' and 'Vessel_Type' columns
     df['Ship_Name'] = df['Ship_Name'].str.strip().str.lower()
     df['Vessel_Type'] = df['Vessel_Type'].str.strip().str.lower()
     ```

   - **Numerical Columns**

     ```python
     # Example: Ensure 'Fuel_Consumption' and 'CO2_Emissions' are numeric
     df['Fuel_Consumption'] = pd.to_numeric(df['Fuel_Consumption'], errors='coerce')
     df['CO2_Emissions'] = pd.to_numeric(df['CO2_Emissions'], errors='coerce')
     ```

   - **Date Columns**

     ```python
     # Example: Convert 'Date' column to datetime format
     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
     ```

7. **Handle Outliers**

   Identify and handle outliers in numerical columns.

   ```python
   # Example: Visualize outliers in 'Fuel_Consumption'
   sns.boxplot(x=df['Fuel_Consumption'])
   plt.show()

   # Remove outliers based on IQR
   Q1 = df['Fuel_Consumption'].quantile(0.25)
   Q3 = df['Fuel_Consumption'].quantile(0.75)
   IQR = Q3 - Q1
   df = df[~((df['Fuel_Consumption'] < (Q1 - 1.5 * IQR)) | (df['Fuel_Consumption'] > (Q3 + 1.5 * IQR)))]
   ```

8. **Standardize Units**

   Convert measurements to consistent units if necessary.

   ```python
   # Example: Convert fuel consumption from gallons to liters
   df['Fuel_Consumption'] = df['Fuel_Consumption'] * 3.78541  # 1 gallon = 3.78541 liters
   ```

9. **Save the Cleaned Dataset**

   ```python
   # Save the cleaned dataset to a new CSV file
   df.to_csv('cleaned_ship_fuel_consumption_co2_emissions.csv', index=False)
   ```

10. **Verify the Cleaned Dataset**

    ```python
    # Check the cleaned dataset
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    ```

### Notes:

- **Replace Column Names:** Ensure that you replace `'Ship_Name'`, `'Vessel_Type'`, `'Fuel_Consumption'`, `'CO2_Emissions'`, and `'Date'` with the actual column names from your dataset.
- **Adjust Cleaning Steps:** Modify the cleaning steps based on the specific characteristics and issues present in your dataset.
- **Error Handling:** If you encounter any errors, check the data types, ensure the file path is correct, and verify that the dataset loads properly.

This code provides a comprehensive approach to cleaning your local dataset. Adjust the steps as needed to address the specific requirements of your data.
