# update_data.py

from datetime import date, timedelta
import os
from APH import fill_data_into_performance_csv

# Vérifier si les données doivent être mises à jour
def should_update_data():
    today = date.today()
    try:
        with open('last_update.txt', 'r') as file:
            last_update = datetime.strptime(file.read().strip(), '%Y-%m-%d').date()
    except FileNotFoundError:
        return True  # Si le fichier n'existe pas, forcer la mise à jour

    return last_update < today

# Mettre à jour la date de la dernière mise à jour
def update_last_update_file():
    with open('last_update.txt', 'w') as file:
        file.write(date.today().strftime('%Y-%m-%d'))

# Mise à jour automatique
if should_update_data():
    print("Mise à jour des données...")
    fill_data_into_performance_csv()
    update_last_update_file()
    print("Mise à jour terminée.")
else:
    print("Les données sont déjà à jour.")
