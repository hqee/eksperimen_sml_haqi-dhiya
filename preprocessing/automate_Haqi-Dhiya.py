import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def run_automation(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' tidak ditemukan.")
        return

    df = pd.read_csv(input_file)
    
    # Menangani Missing Values 
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    for col in cols_with_zeros:
        df[col] = df[col].fillna(df[col].median())
    
    # Capping Outliers 
    cols_to_cap = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    for col in cols_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    
    # Standarisasi
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['Outcome'] = y.values
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"Otomatisasi Berhasil! Data disimpan di: {output_file}")

if __name__ == "__main__":
    input_path = "diabetes.csv" 
    output_path = "preprocessing/diabetes_preprocessed.csv"
    
    run_automation(input_path, output_path)