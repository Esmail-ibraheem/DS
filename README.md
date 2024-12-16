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

---

### with kaggle dataset: 

To clean the dataset from the Kaggle link you provided, we need to follow a systematic approach. Since I cannot directly access the dataset, I will outline a general cleaning process that you can adapt to your specific dataset. You can use Python with libraries like pandas for data cleaning.

Here’s a step-by-step guide to clean the dataset:

---

### Step 1: Load the Dataset
First, load the dataset into a pandas DataFrame.

```python
import pandas as pd

# Load the dataset
url = "https://www.kaggle.com/datasets/jeleeladekunlefijabi/ship-fuel-consumption-and-co2-emissions-analysis"
# Replace with the actual file path if you download the dataset
file_path = "ship_fuel_consumption_co2_emissions.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())
```

---

### Step 2: Inspect the Dataset
Check the structure, missing values, and data types.

```python
# Check the shape of the dataset
print(df.shape)

# Check data types and missing values
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check for duplicate rows
print(df.duplicated().sum())
```

---

### Step 3: Handle Missing Values
Decide how to handle missing values based on the context of the data.

```python
# Drop rows with too many missing values
df.dropna(subset=['column_name'], inplace=True)  # Replace 'column_name' with actual column names

# Fill missing values with appropriate methods
df['column_name'].fillna(df['column_name'].median(), inplace=True)  # For numerical columns
df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)  # For categorical columns
```

---

### Step 4: Handle Duplicates
Remove duplicate rows if they exist.

```python
# Drop duplicate rows
df.drop_duplicates(inplace=True)
```

---

### Step 5: Clean Specific Columns
Depending on the dataset, clean specific columns. For example:

#### 1. **Text Columns (e.g., Ship Name, Vessel Type)**
Remove leading/trailing spaces, convert to lowercase, or standardize values.

```python
# Clean text columns
df['Ship_Name'] = df['Ship_Name'].str.strip().str.lower()
df['Vessel_Type'] = df['Vessel_Type'].str.strip().str.lower()
```

#### 2. **Numerical Columns (e.g., Fuel Consumption, CO2 Emissions)**
Ensure numerical columns are in the correct format.

```python
# Convert columns to numeric
df['Fuel_Consumption'] = pd.to_numeric(df['Fuel_Consumption'], errors='coerce')
df['CO2_Emissions'] = pd.to_numeric(df['CO2_Emissions'], errors='coerce')
```

#### 3. **Date Columns**
If there are date columns, convert them to datetime format.

```python
# Convert date columns to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
```

---

### Step 6: Handle Outliers
Check for outliers in numerical columns and handle them appropriately.

```python
# Check for outliers using boxplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Fuel_Consumption'])
plt.show()

# Remove outliers (optional)
Q1 = df['Fuel_Consumption'].quantile(0.25)
Q3 = df['Fuel_Consumption'].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df['Fuel_Consumption'] < (Q1 - 1.5 * IQR)) | (df['Fuel_Consumption'] > (Q3 + 1.5 * IQR)))]
```

---

### Step 7: Standardize Units
If the dataset contains measurements in different units, standardize them.

```python
# Example: Convert fuel consumption from gallons to liters
df['Fuel_Consumption'] = df['Fuel_Consumption'] * 3.78541  # 1 gallon = 3.78541 liters
```

---

### Step 8: Save the Cleaned Dataset
After cleaning, save the cleaned dataset to a new file.

```python
# Save the cleaned dataset
df.to_csv('cleaned_ship_fuel_consumption_co2_emissions.csv', index=False)
```

---

### Step 9: Verify the Cleaned Dataset
Finally, verify the cleaned dataset to ensure it meets your requirements.

```python
# Check the cleaned dataset
print(df.head())
print(df.info())
print(df.isnull().sum())
```

---

### Notes:
- Replace `'column_name'` with the actual column names in your dataset.
- Adjust the cleaning steps based on the specific issues in your dataset.
- If you encounter any specific errors or issues, let me know, and I can help troubleshoot.

