import subprocess
import pandas as pd
import matplotlib.pyplot as plt
#from intro_script import intro

def get_repo_name():
    # Obtenir la URL del remote
    result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], stdout=subprocess.PIPE)
    url = result.stdout.decode('utf-8').strip()

    # Extreure el nom del repositori de la URL
    repo_name = url.split('/')[-1].replace('.git', '')
    return repo_name


def get_commit_details():
    result = subprocess.run(['git', 'log', '--numstat', '--pretty=format:%H %an %ae %ad', '--date=short', '--no-merges'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    commit_details = []
    current_commit = None

    for line in lines:
        if line:
            print(f'linia[0]: {line[0]}')
        if line and not line[0].isdigit():
            print(f'Current line: {line}')
            parts = line.split()
            commit_hash = parts[0]
            author = " ".join(parts[1:-2])
            email = parts[-2]
            date = parts[-1]
            current_commit = {'hash': commit_hash, 'author': author, 'email': email, 'date': date, 'files': []}
            commit_details.append(current_commit)
        elif line and current_commit:
            print(f'Current line: {line}')
            parts = line.split()
            if len(parts) == 3:
                added, deleted, filename = parts
                current_commit['files'].append({'added': int(added), 'deleted': int(deleted), 'filename': filename})

    # Comprovem que haguem obtingut les dades correctament
    if len(commit_details) == 0:
        # revisem quantes liies s'han recuperat del git log
        print("Línies recuperades:")
        print(lines)
        raise ValueError("No s'han trobat dades de commits.")

    return commit_details

def format_commit_details(commit_details):
    data = []
    for commit in commit_details:
        for file in commit['files']:
            data.append([commit['date'], commit['email'], file['filename'], file['added'], file['deleted']])
    return data

def contribucio_fitxers(df):
    # Calcular el total de línies canviades (afegides + esborrades) per cada fila
    df['Total Lines Changed'] = df['Lines Added'] + df['Lines Deleted']

    # Excloure els fitxers que no hagin estat modificats
    df = df[df['Total Lines Changed'] > 0]

    # Excloure els fitxers de les carpetes ocultes (que comencin amb un .)
    df = df[~df['Filename'].str.startswith('.')]

    # Agrupar per Filename i User ID i calcular la suma de Total Lines Changed
    grouped = df.groupby(['Filename', 'User ID'])['Total Lines Changed'].sum().reset_index()

    # Calcular el total de línies canviades per cada fitxer
    total_lines_per_file = grouped.groupby('Filename')['Total Lines Changed'].sum().reset_index()
    total_lines_per_file = total_lines_per_file.rename(columns={'Total Lines Changed': 'Total Lines Per File'})

    # Unir les dades agrupades amb el total de línies per fitxer
    merged = pd.merge(grouped, total_lines_per_file, on='Filename')

    # Calcular el percentatge de contribució per cada usuari
    merged['Percentage Contribution'] = (merged['Total Lines Changed'] / merged['Total Lines Per File']) * 100

    return merged

def contribucio_per_data(df):
    # Calcular el total de línies canviades (afegides + esborrades) per cada fila
    df['Total Lines Changed'] = df['Lines Added'] + df['Lines Deleted']

    # Excloure els fitxers que no hagin estat modificats
    df = df[df['Total Lines Changed'] > 0]

    # Excloure els fitxers de les carpetes ocultes (que comencin amb un .)
    df = df[~df['Filename'].str.startswith('.')]

    # Agrupar per Data i User ID i calcular la suma de Total Lines Changed
    grouped = df.groupby(['Date', 'User ID'])['Total Lines Changed'].sum().reset_index()

    # Calcular el total de línies canviades per cada data
    total_lines_per_date = grouped.groupby('Date')['Total Lines Changed'].sum().reset_index()
    total_lines_per_date = total_lines_per_date.rename(columns={'Total Lines Changed': 'Total Lines Per Date'})

    # Unir les dades agrupades amb el total de línies per data
    merged = pd.merge(grouped, total_lines_per_date, on='Date')

    # Calcular el percentatge de contribució per cada usuari
    merged['Percentage Contribution'] = (merged['Total Lines Changed'] / merged['Total Lines Per Date']) * 100

    return merged

try:
    commit_details = get_commit_details()
    formatted_data = format_commit_details(commit_details)

    # Create a DataFrame and print it as a table
    df = pd.DataFrame(formatted_data, columns=['Date', 'User ID', 'Filename', 'Lines Added', 'Lines Deleted'])
    print(df.to_string(index=False))


    # Cridar la funció i mostrar el resultat
    resultat_fitxers = contribucio_fitxers(df)
    print(resultat_fitxers)

    # Crear una visualització de columnes apilades
    pivot_df = resultat_fitxers.pivot(index='Filename', columns='User ID', values='Percentage Contribution')
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.xlabel('Filename')
    plt.ylabel('Percentage Contribution')
    plt.title('Percentage Contribution per User ID and Filename')
    plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.savefig('informe_contribucio_stacked.png', bbox_inches='tight')
    plt.show()

    # Cridar la funció i obtenir el resultat
    resultat_per_data = contribucio_per_data(df)
    print(resultat_per_data)


    # Crear una visualització de columnes apilades utilitzant el total en lloc dels percentatges
    pivot_df = resultat_per_data.pivot(index='Date', columns='User ID', values='Total Lines Changed')
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.xlabel('Date')
    plt.ylabel('Total Lines Changed')
    plt.title('Total Lines Changed per User ID and Date')
    plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.savefig('informe_contribucio_total.png', bbox_inches='tight')
    plt.show()


    # Obtenir el número del grup a partir del nom del repositori
    repo_name = get_repo_name()
    group_number = repo_name.split('_')[-1]



    # Generar l'informe HTML assegurant-se que no hi hagi problemes d'accents
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Informe del Grup {group_number}</title>
    </head>
    <body>
        <h1>Informe del Grup {group_number}</h1>
        
        <h2>Dataframe per Fitxer</h2>
        {resultat_fitxers.to_html(index=False)}
    
        <h2>Visualització per Fitxer</h2>
        <img src="informe_contribucio_stacked.png" alt="Informe Contribució per Fitxer">
    
        <h2>Dataframe per Data</h2>
        {resultat_per_data.to_html(index=False)}
    
        <h2>Visualització per Data</h2>
        <img src="informe_contribucio_total.png" alt="Informe Contribució per Data">
    </body>
    </html>
    """

    # Guardar l'informe HTML en un fitxer
    with open('report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Informe HTML generat amb èxit.")
except ValueError as e:
    print(f"Error: {e}")
