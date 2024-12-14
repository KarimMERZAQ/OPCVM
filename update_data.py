import requests
from datetime import datetime, timedelta, date
import pandas as pd
import os

# Fonction pour normaliser les benchmarks
def normalize_benchmark(dataframe):
    reference = {
        "TMI": 1, "Taux Repo": 2, "MBI": 3, "MONIA": 4,
        "Masi": 5, "MASI": 5, "Repo jj": 6, "CFG": 7, "TMP": 8
    }
    dataframe.rename(columns={'Indice Bentchmark': 'Indice Benchmark'}, inplace=True)
    dataframe.rename(columns={'Dénomination OPCVM': 'Info'}, inplace=True)
    dataframe.rename(columns={'OPCVM': 'Info'}, inplace=True)

    if 'Info' not in dataframe.columns:
        print("Error: 'Info' column not found in the dataframe")
        return None
    
    dataframe['Indice Benchmark'] = dataframe['Indice Benchmark'].fillna('').astype(str)
    normalized_names = []
    for fund_type in dataframe['Indice Benchmark']:
        normalized_fund = [key for key in reference.keys() if key in fund_type]
        normalized_names.append('+'.join(normalized_fund) if normalized_fund else None)
    
    dataframe['Normal'] = normalized_names
    dataframe['Info'] = dataframe['Info'].str.strip().str.upper()
    return dataframe

# Fonction pour télécharger et traiter un fichier pour une date donnée
def download_and_process_file(date_to_process):
    if isinstance(date_to_process, str):
        date_to_process = datetime.strptime(date_to_process, "%d-%m-%Y").date()
    formatted_date = date_to_process.strftime("%d-%m-%Y")

    year_month = date_to_process.strftime("%Y/%m")
    previous_month = (date_to_process.replace(day=1) - timedelta(days=1)).strftime("%Y/%m")
    next_month = (date_to_process.replace(day=28) + timedelta(days=4)).strftime("%Y/%m")
    
    url_patterns = [
        f"https://asfim.ma/wp-content/uploads/{year_month}/Tableau-des-Performances-Hebdomadaires-au-",
        f"https://asfim.ma/wp-content/uploads/{year_month}/Tableau-des-performances-quotidiennes-au-",
        f"https://asfim.ma/wp-content/uploads/{previous_month}/Tableau-des-Performances-Hebdomadaires-au-",
        f"https://asfim.ma/wp-content/uploads/{previous_month}/Tableau-des-performances-quotidiennes-au-",
        f"https://asfim.ma/wp-content/uploads/{next_month}/Tableau-des-Performances-Hebdomadaires-au-",
        f"https://asfim.ma/wp-content/uploads/{next_month}/Tableau-des-performances-quotidiennes-au-"
    ]
    
    generated_url = None
    for url_base in url_patterns:
        generated_url = f"{url_base}{formatted_date}.xlsx"
        response = requests.get(generated_url)
        if response.status_code == 200:
            break
    else:
        print(f"No file found for {formatted_date}.")
        return None

    if response.status_code == 200:
        excel_file = f"Tableau_Performances_{formatted_date}.xlsx"
        with open(excel_file, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully: {excel_file}")
        
        df = pd.read_excel(excel_file, skiprows=1)
        os.remove(excel_file)
        
        # Check for empty or malformed data
        if df.empty:
            print("Error: Downloaded file is empty.")
            return None
        
        # Save as CSV
        csv_file = 'TPC.csv'
        df.to_csv(csv_file, index=False)
        
        df_info_fonds = pd.read_csv(csv_file)
        print("Downloaded DataFrame columns:", df_info_fonds.columns)
        
        df_info_fonds.rename(columns={'Dénomination OPCVM': 'Info'}, inplace=True)
        df_info_fonds.rename(columns={'Périodicité VL': 'Periodicite VL'}, inplace=True)

        os.makedirs('DataF', exist_ok=True)  # Ensure directory exists

        # Normalize the data
        df_info_fonds_normalized = normalize_benchmark(df_info_fonds)
        
        if df_info_fonds_normalized is None:
            print("Normalization failed.")
            return None
        
        df_info_fonds_normalized.to_csv('DataF/df_info_fonds_normalized.csv', index=False)
        print(f"Data processed successfully for {formatted_date}.")
        return df_info_fonds_normalized
    else:
        print(f"Error downloading file for {formatted_date}. HTTP Status Code: {response.status_code}")
        return None

# Fonction pour remplir le fichier "df_performance_cleaned.csv" avec les nouvelles données
def fill_data_into_performance_csv():
    df2 = pd.read_csv('DataF/df_performance_cleaned.csv')

    # Ensure the DataFrame is sorted by date in descending order
    df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d')
    df2 = df2.sort_values(by='Date', ascending=False)  # Sort by date, newest first
    df2['Date'] = df2['Date'].dt.strftime('%Y-%m-%d')

    last_date = pd.to_datetime(df2['Date']).max().date()
    df2.columns = df2.columns.str.strip()
    df2.iloc[:, 1:] = df2.iloc[:, 1:].apply(lambda x: x.str.strip().str.upper() if x.dtype == "object" else x)
    
    today = date.today()
    current_date = last_date + timedelta(days=1)
    
    while current_date <= today:
        print(f"Processing data for {current_date}")
        df_info_fonds_normalized = download_and_process_file(current_date)
        if df_info_fonds_normalized is not None:
            df1 = df_info_fonds_normalized.copy()
            df1['Info'] = df1['Info'].str.strip().str.upper()
            string_columns = df2.columns[(df2.dtypes == 'object') & (df2.columns != 'Date')]
            df2[string_columns] = df2[string_columns].apply(lambda x: x.str.upper())
            
            new_row = {'Date': current_date}
            for column in df2.columns[1:]:
                if column in df1['Info'].values:
                    vl_value = df1.loc[df1['Info'] == column, 'VL'].values[0]
                    new_row[column] = vl_value
                else:
                    # Fetch the last non-null value from the most recent records
                    last_non_null_value = df2.loc[df2[column].notnull(), column].iloc[0]
                    new_row[column] = last_non_null_value

                    # If no non-null value is found, log a warning
                    if pd.isnull(new_row[column]):
                        print(f"Warning: No previous value found for {column}")
            
            # Append the new row
            new_row_df = pd.DataFrame([new_row], columns=df2.columns)
            df2 = pd.concat([new_row_df, df2], ignore_index=True)
            print(f"Row processed for {current_date}.")
        current_date += timedelta(days=1)
    
    # Save final DataFrame
    df2.to_csv('DataF/df_performance_cleaned.csv', index=False)
    dt = pd.to_datetime(df2['Date']).max().date()
    return f"Data saved: Latest data date is: {dt}."

# Point d'entrée pour exécuter les mises à jour
if __name__ == "__main__":
    print("Mise à jour des données en cours...")
    latest_date = fill_data_into_performance_csv()
    print(f"Mise à jour terminée : {latest_date}")
