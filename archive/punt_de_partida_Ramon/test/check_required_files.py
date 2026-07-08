import os

def check_required_files(directory, required_files):
    missing_files = [file for file in required_files if not os.path.isfile(os.path.join(directory, file))]
    return missing_files

# Defineix el directori del repositori i els fitxers requerits
repo_directory = '.'  # Canvia aquest valor si el directori és diferent
required_files = ['README.md', 'LICENSE', 'main.py','environment.yml']

# Comprova si els fitxers requerits existeixen
missing_files = check_required_files(repo_directory, required_files)

# Mostra els resultats
if not missing_files:
    print("Tots els fitxers requerits estan presents.")
else:
    print("Falten els següents fitxers requerits:")
    for file in missing_files:
        print(file)