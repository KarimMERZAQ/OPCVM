import requests
from datetime import datetime, timedelta, date
import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
        
        csv_file = 'TPC.csv'
        df.to_csv(csv_file, index=False)
        if 'Hebdomadaires' in generated_url:
            df_info_fonds_normalized.to_csv('DataF/df_info_fonds_normalized.csv', index=False)
        
        df_info_fonds = pd.read_csv(csv_file)
        df_info_fonds.rename(columns={'Dénomination OPCVM': 'Info'}, inplace=True)
        df_info_fonds.rename(columns={'Périodicité VL': 'Periodicite VL'}, inplace=True)
        
        print("Columns in the downloaded file:", df_info_fonds.columns)
        df_info_fonds_normalized = normalize_benchmark(df_info_fonds)
        
        if df_info_fonds_normalized is None:
            print("Normalization failed due to missing 'Info' column.")
            return None
        
        
        print(f"Data processed for {formatted_date}.")
        return df_info_fonds_normalized

    print(f"No file found for {formatted_date}.")
    return None

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
    #os.remove('TPC.csv')
    return(f"Data saved: Latest data date is: {dt}.")

def MAJ_DATA():
    st.write(fill_data_into_performance_csv())

if st.button('Mettre les données à jour'):
    MAJ_DATA()

def filter_comparable_fundsFirst(df, benchmark, periodicity, classification):
   
    comparable_funds = df.copy()

    if benchmark != 'Tout':
        if 'Normal' in df.columns:
            comparable_funds = comparable_funds[comparable_funds['Normal'] == benchmark]
        else:
            raise KeyError("'Normal' column not found in DataFrame")

    if classification != 'Tout':
        if 'Classification' in df.columns:
            comparable_funds = comparable_funds[comparable_funds['Classification'] == classification]
        else:
            raise KeyError("'Classification' column not found in DataFrame")

    if periodicity != 'Tout':
        if 'Periodicite VL' in df.columns:
            comparable_funds = comparable_funds[comparable_funds['Periodicite VL'] == periodicity]
        else:
            raise KeyError("'Periodicite VL' column not found in DataFrame")

  
    if 'Info' in comparable_funds.columns:
        return comparable_funds['Info'].tolist()
    else:
        raise KeyError("'Info' column not found in filtered DataFrame")

# Load the cleaned data from CSV files
df_performance_cleaned = pd.read_csv("DataF/df_performance_cleaned.csv", parse_dates=['Date'])
df_info_fonds_normalized = pd.read_csv("DataF/df_info_fonds_normalized.csv")

# Application title
st.title("Tableau de Bord des Fonds OPCVM")

# Sidebar criteria selection
st.sidebar.markdown("# Choisissez les critères de comparaison:")
criteria_benchmark = st.sidebar.checkbox('Benchmark', value=True)
criteria_periodicity = st.sidebar.checkbox('Périodicité VL', value=True)
criteria_classification = st.sidebar.checkbox('Classification', value=True)
st.sidebar.write("Comparer le fond seleéctionné avec les fonds qui ont un quartile :")
choix = st.sidebar.selectbox(
    "Sélectionnez une option :",
    ["75%", "50%", "25%", "Ne pas prendre actif net en considération"]
)

benchs = df_info_fonds_normalized['Normal'].str.upper().dropna().unique().tolist()
benchs = [str(b) for b in benchs]
optionsbench = ["Tout"] + benchs 

clsfs = df_info_fonds_normalized['Classification'].dropna().unique().tolist()
clsfs = [str(p) for p in clsfs]
optionsclsf = ["Tout"] + clsfs  

bench = st.selectbox("Sélectionnez  indice de benchmark :", optionsbench)
prd = st.selectbox("Sélectionnez la périodicité :", ["Tout" ,"Hebdomadaire", "Quotidienne"])
clsf = st.selectbox("Sélectionnez la classification :", optionsclsf)

# Filter funds based on selected criteria
funds = filter_comparable_fundsFirst(df_info_fonds_normalized, bench, prd, clsf)
selected_fund = st.selectbox(f"Sélectionner un fond:({len(funds)} fonds)", funds)






#Periode
dates = df_performance_cleaned['Date'].dropna().sort_values().unique()
date_options = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in dates]

start_date = st.selectbox('Choisissez la date de début', options=date_options)
filtered_end_date_options = sorted(
    [date for date in date_options if date > start_date],
    reverse=True
)

# Sélecteur pour end_date
end_date = st.selectbox('Choisissez la date de fin', options=filtered_end_date_options)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)


if start_date is not None and end_date is not None:
    df_performance_cleaned = df_performance_cleaned[(df_performance_cleaned['Date'] >= start_date) & (df_performance_cleaned['Date'] <= end_date)]

# Choix de l'onglet de visualisation
tab = st.selectbox("Choisir une visualisation:", ['Performance Historique', 'Volatilité vs Rendement', 'Comparaison Directe(Perforamance historique)','Comparaison Directe(Volatilité_Rendement)','Générer Rapport PDF'])


def filter_comparable_fundsFirst(df,benchmark,periodicity,classification):
  
    filter_conditions = [df['Info']]

    #fund_actif_net = df.loc[df['Info'],'AN'].values[0]
    
    #actif_net_values = df['AN'].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Critères existants
    if benchmark != 'Tout':
        filter_conditions.append(df['Normal'] == benchmark)
        
    if classification != 'Tout':
        filter_conditions.append(df['Classification'] == classification)    
    
    if periodicity != 'Tout':
        filter_conditions.append(df['Periodicite VL'] == periodicity)
        
    

    # Create a DataFrame with the 'Info' and 'AN' columns
    #actif_net_df = df[['Info', 'AN']].copy()

    # Convert 'AN' to numeric, coercing errors to NaN
    #actif_net_df['AN'] = pd.to_numeric(actif_net_df['AN'], errors='coerce')

    # Drop rows where 'AN' is NaN
    #actif_net_values = actif_net_df.dropna(subset=['AN'])

          #     Initialize the list to store filter conditions
    


    # Start with the full DataFrame and apply all conditions
    comparable_funds = df
    for condition in filter_conditions:
        comparable_funds = comparable_funds[condition]
    
    # Re    turn the filtered list of comparable funds' 'Info'
    return comparable_funds['Info'].tolist()

def filter_comparable_funds(df,fund_name):
    fund_info = df[df['Info'] == fund_name]
    if fund_info.empty:
        raise ValueError(f"Le fond {fund_name} n'existe pas dans les données.")
    filter_conditions = [df['Info'] != fund_name]

    fund_actif_net = df.loc[df['Info'] == fund_name,'AN'].values[0]
    
    #actif_net_values = df['AN'].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Critères existants
    if criteria_benchmark:
        benchmark = fund_info['Normal'].values[0]
        filter_conditions.append(df['Normal'] == benchmark)
        
    if criteria_classification:
        classification = fund_info['Classification'].values[0]
        filter_conditions.append(df['Classification'] == classification)    
    
    if criteria_periodicity:
        periodicity = fund_info['Periodicite VL'].values[0]
        filter_conditions.append(df['Periodicite VL'] == periodicity)
        
    

    # Create a DataFrame with the 'Info' and 'AN' columns
    actif_net_df = df[['Info', 'AN']].copy()

    # Convert 'AN' to numeric, coercing errors to NaN
    actif_net_df['AN'] = pd.to_numeric(actif_net_df['AN'], errors='coerce')

    # Drop rows where 'AN' is NaN
    actif_net_values = actif_net_df.dropna(subset=['AN'])

          #     Initialize the list to store filter conditions
    

    # Handle different quartile selections based on 'choix'
    if choix == '75%':
        quartile_75 = actif_net_values['AN'].quantile(0.75)
        if fund_actif_net <= quartile_75:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] <= quartile_75]['Info']))
        else:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] > quartile_75]['Info']))

    elif choix == '50%':
        quartile_50 = actif_net_values['AN'].quantile(0.50)
        if fund_actif_net <= quartile_50:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] <= quartile_50]['Info']))
        else:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] > quartile_50]['Info']))

    elif choix == '25%':
        quartile_25 = actif_net_values['AN'].quantile(0.25)
        if fund_actif_net <= quartile_25:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] <= quartile_25]['Info']))
        else:
            filter_conditions.append(df['Info'].isin(actif_net_values[actif_net_df['AN'] > quartile_25]['Info']))

    # Start with the full DataFrame and apply all conditions
    comparable_funds = df
    for condition in filter_conditions:
        comparable_funds = comparable_funds[condition]
    
    # Re    turn the filtered list of comparable funds' 'Info'
    return comparable_funds['Info'].tolist()
 
def generate_report(column_name, df_filtered, performance_stats, plot_path, df_info_fonds_normalized):
    if not os.path.exists(plot_path):
        raise FileNotFoundError(f"Le fichier {plot_path} n'existe pas.")
    
    # Récupérer les informations spécifiques du fonds à partir des DataFrames déjà chargés
    NJ = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == column_name, 'Nature juridique'].values[0]
    CS = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == column_name, 'Classification'].values[0]
    IB = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == column_name, 'Indice Benchmark'].values[0]
    PV = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == column_name, 'Periodicite VL'].values[0]
    MAN = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == column_name, 'AN'].values[0]
    E = 'un' if PV == 'Hebdomadaire' else 'une'
    
    pdf_path = f"reports/{column_name}_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    # Titre du rapport
    flowables.append(Paragraph(f"Reporting {column_name}", styles['Title']))
    flowables.append(Spacer(1, 12))

    # Introduction du fonds
    intro_text = f"""
    <b>Le fonds {column_name}</b> est {E} {NJ} {CS} {PV}, dont l’indice de Benchmark est {IB}, et dont l’actif net actuel est de {MAN} DH.
    """
    flowables.append(Paragraph(intro_text, styles['BodyText']))
    flowables.append(Spacer(1, 12))

    # Ajout du graphique de performance historique
    flowables.append(Paragraph("Performances Historiques", styles['Heading2']))
    flowables.append(Image(plot_path, width=500, height=300))
    flowables.append(Spacer(1, 12))

    # Ajout des performances journalières
    performance_text = f"""
    <b>Statistiques de Performance Journalière</b><br/><br/>
    - Jours de performance positive : {performance_stats['positive_days']}<br/><br/>
    - Jours de performance négative : {performance_stats['negative_days']}<br/><br/>
    - Nombre total de jours : {performance_stats['total_days']}<br/><br/>
    - La performance journalière moyenne observée est : {performance_stats['mean']:.4%}<br/><br/>
    - La performance journalière positive maximale observée : {performance_stats['max_positive']:.4%}<br/><br/>
    - La performance journalière négative maximale observée  : {performance_stats['max_negative']:.4%}
    """
    
    flowables.append(Paragraph(performance_text, styles['BodyText']))
    flowables.append(Spacer(1, 12))

    # Génération du fichier PDF
    doc.build(flowables)
    return pdf_path


def generate_performance_plot(fund, initial_investment):
    # Vérifier si la colonne 'Date' existe
    if 'Date' not in df_performance_cleaned.columns:
        raise KeyError("La colonne 'Date' est manquante dans le DataFrame.")
    
    # Filtrer les données
    df_filtered = df_performance_cleaned[['Date', fund]].dropna()

    # Convertir la colonne 'Date' en datetime si nécessaire
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

    # Inverser l'ordre des lignes pour avoir la plus ancienne date en bas
    df_filtered = df_filtered.iloc[::-1].reset_index(drop=True)

    # Ajouter la colonne normalisée (VLN)
    first_value = df_filtered[fund].iloc[0]  # Première valeur du fond (la plus ancienne)
    df_filtered['VLN'] = (df_filtered[fund] / first_value) * 100  # Normalisation par rapport à la première valeur

    # Récupérer la dernière date et la dernière valeur de VLN
    last_date = df_filtered['Date'].iloc[-1]
    last_vln = df_filtered['VLN'].iloc[-1]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer la valeur normalisée (VLN)
    df_filtered.plot(x='Date', y='VLN', ax=ax, label='Valeur Normalisée (VLN)', color='green')

    # Ajouter des titres et des labels
    ax.set_title(f'{fund} - Valeur Normalisée (VLN)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur Normalisée (VLN)')
    ax.grid(True)

    # Annoter la dernière valeur de VLN sur le graphique
    ax.annotate(f'{fund}: {last_vln:.2f}', xy=(last_date, last_vln), xytext=(last_date, last_vln + 5),
                arrowprops=dict(facecolor='black', shrink=0.05), color='green')

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder le graphique dans le dossier 'assets'
    plot_path = f'assets/{fund}_normalized_value.png'
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path



def generate_performance_plotwait(fund, initial_investment):
    # Vérifier si la colonne 'Date' existe
    if 'Date' not in df_performance_cleaned.columns:
        raise KeyError("La colonne 'Date' est manquante dans le DataFrame.")
    
    # Filtrer les données et calculer la performance (V1 - V0) / V0
    df_filtered = df_performance_cleaned[['Date', fund]].dropna()

    # Convertir la colonne 'Date' en datetime si nécessaire
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

    # Inverser l'ordre des lignes car la première ligne correspond à la date la plus récente
    df_filtered = df_filtered.iloc[::-1].reset_index(drop=True)

    # Calcul de la performance
    df_filtered['Performance'] = (df_filtered[fund] - df_filtered[fund].shift(1)) / df_filtered[fund].shift(1)
    
    # Multiplier la performance par 100 pour augmenter l'effet (si les variations sont trop petites)
    df_filtered['Performance'] = df_filtered['Performance'] * 100

    # Supprimer les valeurs NaN résultant du calcul
    df_filtered = df_filtered.dropna(subset=['Performance'])

    # Utiliser 'Date' comme index pour le resampling
    df_filtered = df_filtered.set_index('Date')

    # Resample les données pour obtenir des valeurs mensuelles
    df_resampled = df_filtered.resample('M').mean()

    # Reset l'index pour remettre la colonne 'Date' en tant que colonne régulière
    df_resampled = df_resampled.reset_index()

    # Calculer la valeur cumulée du portefeuille en partant de l'investissement initial
    df_resampled['Portfolio Value'] = initial_investment * (1 + df_resampled['Performance'] / 100).cumprod()

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    df_resampled.plot(x='Date', y='Portfolio Value', ax=ax, label='Valeur du portefeuille', color='blue')

    # Ajouter des titres et des labels
    ax.set_title(f' {fund} (Investissement initial = {initial_investment} DH)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur du portefeuille (DH)')
    ax.grid(True)

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder le graphique dans le dossier 'assets'
    plot_path = f'assets/{fund}_portfolio_value.png'
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path

def generate_volatility_vs_return_plot(fund,P):
    df_subset = df_performance_cleaned[['Date', fund]].dropna()
    
   
    if P == 'Mois':
        # Resampling hebdomadaire pour obtenir les variations hebdomadaires
        df_weekly = df_subset.set_index('Date').resample('M').last()
        df_weekly_changes = df_weekly.pct_change().dropna()    

        fig, ax = plt.subplots(figsize=(14, 8))

        # Boucle sur les semaines et affichage de chaque point
        for i, (week, data) in enumerate(df_weekly_changes.iterrows()):
            weekly_return = data[fund]
            volatility = df_weekly_changes[:week].std()[fund]
        
        # Ajouter chaque point au graphique avec une annotation pour la semaine
            ax.scatter(volatility, weekly_return, label=f'{P} {i+1}', alpha=0.6)
            if i % 4 == 0:  # Annotation toutes les 4 semaines
                ax.annotate(f'{P} {i+1}', (volatility, weekly_return), textcoords="offset points", xytext=(5, 5), ha='center')
    else:
         # Resampling hebdomadaire pour obtenir les variations hebdomadaires
        df_weekly = df_subset.set_index('Date').resample('W').last()
        df_weekly_changes = df_weekly.pct_change().dropna()    

        fig, ax = plt.subplots(figsize=(14, 8))

        # Boucle sur les semaines et affichage de chaque point
        for i, (week, data) in enumerate(df_weekly_changes.iterrows()):
            weekly_return = data[fund]
            volatility = df_weekly_changes[:week].std()[fund]
        
        # Ajouter chaque point au graphique avec une annotation pour la semaine
            ax.scatter(volatility, weekly_return, label=f'{P} {i+1}', alpha=0.6)
            if i % 4 == 0:  # Annotation toutes les 4 semaines
                ax.annotate(f'{P} {i+1}', (volatility, weekly_return), textcoords="offset points", xytext=(5, 5), ha='center')

    plt.xlabel('Volatilité (Écart-type des variations hebdomadaires)')
    plt.ylabel('Rendement hebdomadaire')
    plt.title(f'Rendement vs Volatilité du fond {fund} sur plusieurs {P}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=P)
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarder le graphique
    plot_path = f'assets/{fund}_weekly_volatility_vs_return.png'
    plt.savefig(plot_path)
    plt.close(fig)
    
    return plot_path



def generate_comparison_plot(fund1, fund2, initial_investment):
    # Vérifier si la colonne 'Date' existe
    if 'Date' not in df_performance_cleaned.columns:
        raise KeyError("La colonne 'Date' est manquante dans le DataFrame.")
    
    # Filtrer les données pour les fonds spécifiés
    df_filtered = df_performance_cleaned[['Date', fund1, fund2]].dropna()

    # Convertir la colonne 'Date' en datetime si nécessaire
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

    # Inverser l'ordre des lignes pour avoir la plus ancienne date en bas
    df_filtered = df_filtered.iloc[::-1].reset_index(drop=True)

    # Ajouter les colonnes normalisées (VLN1 et VLN2)
    first_value1 = df_filtered[fund1].iloc[0]  # Première valeur du fond 1 (plus ancienne)
    first_value2 = df_filtered[fund2].iloc[0]  # Première valeur du fond 2 (plus ancienne)
    
    df_filtered['VLN1'] = (df_filtered[fund1] / first_value1) * 100  # Normaliser fund1
    df_filtered['VLN2'] = (df_filtered[fund2] / first_value2) * 100  # Normaliser fund2

    # Récupérer la dernière date et la dernière valeur de chaque VLN
    last_date = df_filtered['Date'].iloc[-1]
    last_vln1 = df_filtered['VLN1'].iloc[-1]
    last_vln2 = df_filtered['VLN2'].iloc[-1]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les valeurs normalisées (VLN1 et VLN2)
    df_filtered.plot(x='Date', y='VLN1', ax=ax, label=fund1, color='red')
    df_filtered.plot(x='Date', y='VLN2', ax=ax, label=fund2, color='green')

    # Ajouter des titres et des labels
    ax.set_title(f'{fund1} et {fund2} - Valeur Normalisée (VLN)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur Normalisée (VLN)')
    ax.grid(True)

    # Annoter la dernière valeur de chaque fond sur le graphique
    ax.annotate(f'{fund1}: {last_vln1:.2f}', xy=(last_date, last_vln1), xytext=(last_date, last_vln1 + 5),
                arrowprops=dict(facecolor='black', shrink=0.05), color='red')

    ax.annotate(f'{fund2}: {last_vln2:.2f}', xy=(last_date, last_vln2), xytext=(last_date, last_vln2 + 5),
                arrowprops=dict(facecolor='black', shrink=0.05), color='green')

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder le graphique dans le dossier 'assets'
    plot_path = f'assets/{fund1}_vs_{fund2}_normalized_value.png'
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path



def generate_comparison_plotwait(fund1,fund2, initial_investment):
    # Vérifier si la colonne 'Date' existe
    if 'Date' not in df_performance_cleaned.columns:
        raise KeyError("La colonne 'Date' est manquante dans le DataFrame.")
    
    # Filtrer les données et calculer la performance (V1 - V0) / V0
    df_filtered = df_performance_cleaned[['Date', fund1 , fund2]].dropna()

    # Convertir la colonne 'Date' en datetime si nécessaire
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

    # Inverser l'ordre des lignes car la première ligne correspond à la date la plus récente
    df_filtered = df_filtered.iloc[::-1].reset_index(drop=True)

    # Calcul de la performance
    df_filtered['Performance1'] = (df_filtered[fund1] - df_filtered[fund1].shift(1)) / df_filtered[fund1].shift(1)
    df_filtered['Performance2'] = (df_filtered[fund2] - df_filtered[fund2].shift(1)) / df_filtered[fund2].shift(1)

    # Multiplier la performance par 100 pour augmenter l'effet (si les variations sont trop petites)
    df_filtered['Performance1'] = df_filtered['Performance1'] * 100
    df_filtered['Performance2'] = df_filtered['Performance2'] * 100

    # Supprimer les valeurs NaN résultant du calcul
    df_filtered = df_filtered.dropna(subset=['Performance1'])
    df_filtered = df_filtered.dropna(subset=['Performance2'])

    # Resample les données pour obtenir des valeurs mensuelles
    df_resampled = df_filtered.resample('M', on='Date').mean()

    # Reset l'index pour remettre la colonne 'Date' en tant que colonne régulière
    df_resampled = df_resampled.reset_index()

    # Calculer la valeur cumulée du portefeuille en partant de l'investissement initial
    df_resampled['Portfolio Value1'] = initial_investment * (1 + df_resampled['Performance1']).cumprod()
    df_resampled['Portfolio Value2'] = initial_investment * (1 + df_resampled['Performance2']).cumprod()
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    df_resampled.plot(x='Date', y='Portfolio Value1', ax=ax, label=fund1, color='red')
    df_resampled.plot(x='Date', y='Portfolio Value2', ax=ax, label=fund2, color='green')
    # Ajouter des titres et des labels
    ax.set_title(f'{fund1} et {fund2} (Investissement initial = {initial_investment} DH)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur du portefeuille (DH)')
    ax.grid(True)

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder le graphique dans le dossier 'assets'
    plot_path = f'assets/{fund1}_portfolio_value.png'
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path

def generate_volatility_vs_return_plot2(fund,fund2):
    #comparable_funds = filter_comparable_funds(df_info_fonds_normalized, df_actif_net, fund)
    #comparable_funds.append(fund)
    df_subset = df_performance_cleaned[['Date'] + [cf for cf in [fund,fund2] if cf in df_performance_cleaned.columns]].dropna()
    df_daily_changes = df_subset.set_index('Date').pct_change().dropna()
    volatility = df_daily_changes.std()
    annual_returns = df_subset.set_index('Date').resample('Y').last().pct_change().mean()

    fig, ax = plt.subplots(figsize=(14, 8))
    #for cf in [fund,fund2]:
    #if fund2 in volatility.index:
    ax.scatter(volatility[fund2], annual_returns[fund2], label=fund2, s=100)
    ax.scatter(volatility[fund], annual_returns[fund], color='red', s=100, label=fund)
    plt.xlabel('Volatilité (Écart-type des variations journalières)')
    plt.ylabel('Rendement moyen annuel')
    plt.title('Rendement vs Volatilité des fonds')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Fonds")
    plt.grid(True)
    plt.tight_layout()
    plot_path = f'assets/{fund}_volatility_vs_return.png'
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path

def calculate_performance(df, fund):
    """
    Calcule des statistiques de performance basées sur les données du fonds sélectionné.
    :param df: DataFrame contenant les performances du fonds.
    :param fund: Nom du fonds sélectionné.
    :param df_actif_net: DataFrame contenant les actifs nets.
    :return: Dictionnaire avec les statistiques de performance.
    """
    # Extraction des données du fonds
    df_filtered = df[['Date', fund]].dropna()
    
    # Trier les données par date dans l'ordre croissant (les plus anciennes en premier)
    df_filtered = df_filtered.sort_values(by='Date', ascending=True)
    
    # Calcul des variations journalières (returns)
    df_filtered['daily_return'] = df_filtered[fund].pct_change().dropna()
    
    # Statistiques de performance
    positive_days = (df_filtered['daily_return'] > 0).sum()
    negative_days = (df_filtered['daily_return'] < 0).sum()
    total_days = df_filtered['daily_return'].count()
    mean_return = df_filtered['daily_return'].mean()
    max_positive = df_filtered['daily_return'].max()
    max_negative = df_filtered['daily_return'].min()
    
    # Inversion des valeurs pour corriger le calcul
    mean_return = abs(mean_return)
    max_positive = abs(max_positive)
    max_negative = -abs(max_negative)
    #st.write(df_info_fonds_normalized.columns)
    # Extraction de l'actif net actuel
    #actif_net = df.loc[df['Info'] == fund, 'AN'].values[0]  # if fund in df_actif_net.columns else "N/A"
    actif_net = df_info_fonds_normalized.loc[df_info_fonds_normalized['Info'] == fund, 'AN'].values[0]
    #actif_net = df_actif_net[fund].values[0] if fund in df_actif_net.columns else "N/A"
    

    # Compilation des statistiques
    performance_stats = {
        'actif_net': actif_net,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'total_days': total_days,
        'mean': mean_return,
        'max_positive': max_positive,
        'max_negative': max_negative
    }
    
    return performance_stats

def generate_pdf_report(fund):
    try:
        # Calcul des statistiques de performance
        performance_stats = calculate_performance(df_performance_cleaned, fund)
        
        # Génération du graphique de performance
        plot_path = generate_performance_plot(fund,initial_investment)
        
        # Génération du rapport PDF
        pdf_path = generate_report(fund, df_performance_cleaned, performance_stats, plot_path,df_info_fonds_normalized)
        
        # Lire le fichier PDF pour l'offrir en téléchargement
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        # Retourner les données du PDF pour le bouton de téléchargement
        return pdf_data, pdf_path
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération du rapport PDF : {str(e)}")
        raise

def info_font(df,fund):
    fund_info = df[df['Info'] == fund]
    benchmark = fund_info['Normal'].values[0]
    periodicity = fund_info['Periodicite VL'].values[0]
    classification = fund_info['Classification'].values[0]

    actif_net_value = df.loc[df['Info'] == fund,'AN'].values[0]

    #actif_net_values = df_actif_net.iloc[0].apply(pd.to_numeric, errors='coerce').dropna()
    #fund_actif_net = actif_net_values[fund]
    
    st.sidebar.write(f"Benchmark : {benchmark}")
    st.sidebar.write(f"Priodicité : {periodicity}")
    st.sidebar.write(f"Classification : {classification}")
    st.sidebar.write(f"Actif net : {actif_net_value} DHS")

st.sidebar.markdown(f"### Caractéristiques du fond {selected_fund}:")

info_font(df_info_fonds_normalized, selected_fund)

if tab == 'Performance Historique':
    initial_investment = st.number_input("Entrez un montant en DHS :", min_value=100.0, value=100.0, step=100.0)
    st.write("Performance Historique")
    img_path = generate_performance_plot(selected_fund,initial_investment)
    st.image(img_path)
    
    with open(img_path, "rb") as file:
        img_data = file.read()
    
    st.download_button(
        label="Télécharger l'image de performance",
        data=img_data,
        file_name=f"{selected_fund}_performance.png",
        mime="image/png"
    )

elif tab == 'Volatilité vs Rendement':
    A = st.selectbox('Evolution du fonds dans les :', ['Mois','Semaine'])
    st.write("Volatilité vs Rendement")
    img_path = generate_volatility_vs_return_plot(selected_fund,A)
    st.image(img_path)
    with open(img_path, "rb") as file:
        img_data = file.read()
    st.download_button(
        label="Télécharger l'image Volatilité vs Rendement",
        data=img_data,
        file_name=f"{selected_fund}_volatility_vs_return.png",
        mime="image/png"
    )

elif tab == 'Comparaison Directe(Perforamance historique)':
    comparable_funds = filter_comparable_funds(df_info_fonds_normalized, selected_fund)
    initial_investment = st.number_input("Entrez un montant en DHS :", min_value=100.0, value=100.0, step=100.0)
    fund_2 = st.selectbox(f"Sélectionnez le deuxième fond à comparer: ({len(comparable_funds)} fonds)", [f for f in comparable_funds if f != selected_fund])
    if fund_2:
        img_path = generate_comparison_plot(selected_fund, fund_2, initial_investment)
        st.image(img_path)
            
        with open(img_path, "rb") as file:
            img_data = file.read()
            
        st.download_button(
            label="Télécharger l'image Comparaison Directe",
            data=img_data,
            file_name=f"{selected_fund}_comparaison_directeP.png",
            mime="image/png"
            )
    else:
        st.error("Veuillez sélectionner un deuxième fond pour comparer.")

elif tab == 'Comparaison Directe(Perforamance historique)a':
    st.write("Perforamance historique")
    comparable_funds = filter_comparable_funds(df_info_fonds_normalized, selected_fund)
    initial_investment = st.number_input("Entrez un montant en DHS :", min_value=100.0, value=100.0, step=100.0)
    fund_2 = st.selectbox("Sélectionnez le deuxième fond à comparer:", [f for f in comparable_funds if f != selected_fund])
    img_path = generate_comparison_plot(selected_fund, fund_2,initial_investment)
    st.image(img_path)
    with open(img_path, "rb") as file:
        img_data = file.read()
    
    st.download_button(
        label="Télécharger l'image Comparaison Directe",
        data=img_data,
        file_name=f"{selected_fund}_comparaison_directeP.png",
        mime="image/png"
    )

elif tab == 'Comparaison Directe(Volatilité_Rendement)':
    st.write("Volatilité_Rendement")
    comparable_funds = filter_comparable_funds(df_info_fonds_normalized, selected_fund)
    fund_2 = st.selectbox("Sélectionnez le deuxième fond à comparer:", [f for f in comparable_funds if f != selected_fund])
    img_path = generate_volatility_vs_return_plot2(selected_fund, fund_2)
    st.image(img_path)
    with open(img_path, "rb") as file:
        img_data = file.read()
    st.download_button(
        label="Télécharger l'image Comparaison Directe",
        data=img_data,
        file_name=f"{selected_fund}_comparaison_directeVR.png",
        mime="image/png"
    )

elif tab == 'Générer Rapport PDF':
    initial_investment = st.number_input("Entrez un montant en DHS :", min_value=100.0, value=100.0, step=100.0)
    if st.button("Générer le rapport PDF"):
        pdf_data, pdf_path = generate_pdf_report(selected_fund)
        st.download_button(
            label="Télécharger le rapport PDF",
            data=pdf_data,
            file_name=os.path.basename(pdf_path),
            mime='application/pdf'
        )