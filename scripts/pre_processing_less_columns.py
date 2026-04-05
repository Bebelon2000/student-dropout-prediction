import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import os
import openpyxl

# Carregar o dataset
data = pd.read_csv('data/students_dropout.data')

# Mudar variáveis string para numérico
#data['Nacionality'] = data['Nacionality'].map({'Other': 0, 'Portuguese': 1})
#data['Relocated'] = data['Relocated'].map({'No': 0, 'Yes': 1})
#data['SpecialNeeds'] = data['SpecialNeeds'].map({'No': 0, 'Yes': 1})
#data['HasDebt'] = data['HasDebt'].map({'No': 0, 'Yes': 1})
#data['PayTuition'] = data['PayTuition'].map({'No': 0, 'Yes': 1})
#data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
#data['HasScholarship'] = data['HasScholarship'].map({'No': 0, 'Yes': 1})
#data['Relationship'] = data['Relationship'].map({'Divorced': 0, 'Facto Union': 1, 'Legally Separated': 2, 'Married': 3, 'Single': 4, 'Widower': 5})
#data['MothersHigherEducation'] = data['MothersHigherEducation'].map({'No': 0, 'Yes': 1})
#data['FathersHigherEducation'] = data['FathersHigherEducation'].map({'No': 0, 'Yes': 1})
#data['Dropout'] = data['Dropout'].map({'No': 0, 'Yes': 1})

# Codificar variáveis categóricas
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Contar número inicial de linhas
initial_rows = data.shape[0]

# Remover linhas duplicadas
data = data.drop_duplicates()
after_duplicates_removed = data.shape[0]

# Remover linhas vazias
data = data.dropna()
final_rows = data.shape[0]

# Exibir quantidade de linhas removidas
removed_duplicates = initial_rows - after_duplicates_removed
removed_na = after_duplicates_removed - final_rows
total_removed = removed_duplicates + removed_na

print(f'Linhas duplicadas removidas: {removed_duplicates}')
print(f'Linhas vazias removidas: {removed_na}')
print(f'Total de linhas removidas: {total_removed}')

# Verificar se o dataset está vazio após a remoção de linhas duplicadas e vazias
if data.empty:
    print("O dataset está vazio após a remoção de linhas duplicadas e vazias.")
else:
    # Calcular a matriz de correlação
    corr_matrix = data.corr()
    corr_matrix.to_excel("pre_processed/students_dropout_correlation.xlsx")

    # Encontrar colunas com correlação menor que 0.1
    low_corr_cols = corr_matrix.columns[abs(corr_matrix['Dropout']) < 0.1]

    # Remover essas colunas
    data.drop(columns=low_corr_cols, inplace=True)
    data.to_excel("pre_processed/students_dropout_less_correlations.xlsx")

    # Exibir colunas removidas
    print(f"Colunas removidas devido à correlação baixa ({low_corr_cols.size} colunas):")
    print(low_corr_cols)

    # Salvar dados tratados
    data.to_csv('pre_processed/students_dropout_less_correlation.csv', index=False)
    print('Dados tratados salvos com sucesso.')

    # Exibir informações do dataset
    print('Informações do dataset:')
    print(data.describe(include="all"))

    # Separar features e target
    X = data.drop(columns=['Dropout'])  # Features
    y = data['Dropout']  # Target

    # Separar em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

    # Colocar os dados numéricos na mesma escala
    scaler = MinMaxScaler((-1, 1)).fit(X_train)
    joblib.dump(scaler, 'models/students_dropout_less_correl_scaler.pkl')

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    train = pd.concat([X_train_scaled, y_train], axis='columns', join='inner')
    test = pd.concat([X_test_scaled, y_test], axis='columns', join='inner')

    # Salvar em CSV os dados de treino e teste
    train.to_csv('pre_processed/students_dropout_less_correl_train.csv', index=False)
    test.to_csv('pre_processed/students_dropout_less_correl_test.csv', index=False)

    print("Processamento concluído! Os dados de treino e teste foram salvos com sucesso.")
