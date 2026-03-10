# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:31:30 2025

@author: Carolina Ribeiro
"""

#%% Importando as bibliotecas

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

#%% Importando Banco de dados

df = pd.read_csv('amostras_indices_espectrais.csv')
df.head()

indices = ['NDVI','MNDWI','NDTI','SAVI','NDRE']
classe = 'class'

df_subset = df[indices + [classe]]
df_subset

#%%

classe_map = {0:'Floresta', 1: 'Agua', 2:'Garimpo',3:'Vegetacao_rasa'}
df['classe_nome']=df['class'].map(classe_map)


#Definindo as variáveis
X =df[['NDVI', 'NDTI', 'MNDWI', 'SAVI', 'NDRE']]

y =df['class']

#%%  Dividir os dados em conjuntos de treino e teste
#stratify=y garante que a proporção entre as classes seja mantida

X_train, X_test, y_train, y_test = train_test_split(
    X,y,stratify=y,test_size=0.3,random_state=42
    )

# Contagens absolutas
train_counts = pd.Series(y_train).value_counts().rename(index=classe_map)
test_counts = pd.Series(y_test).value_counts().rename(index=classe_map)

# Contagens relativas (%)
train_pct = (train_counts / len(y_train) * 100).round(2)
test_pct = (test_counts / len(y_test) * 100).round(2)

# Tabela comparativa
split_check = pd.DataFrame({
    'Treino (n)': train_counts,
    'Treino (%)': train_pct,
    'Teste (n)': test_counts,
    'Teste (%)': test_pct
})

print(split_check)


#%% Tunning do modelo - k-folg com GridSearchCV e oversampling SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

oversample = SMOTE(sampling_strategy='minority',random_state=42, k_neighbors=5)
rf = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[('oversample', oversample),
                           ('model', rf)])
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2']
   # 'model__criterion': ['gini', 'entropy'] 
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train.values.ravel())

print("Melhores hiperparâmetros:", grid_search.best_params_)
print("Melhor score (F1-macro):", grid_search.best_score_)

#%% Modelo com os parâmetros para evitar overfitting com SMOTE
oversample = SMOTE(sampling_strategy='minority',random_state=42, k_neighbors=5)

rf_tun = RandomForestClassifier(
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
   )
rf_tun_final = Pipeline(steps=[('oversample', oversample),
                           ('model', rf_tun)])

rf_tun_final.fit(X_train, y_train)
rf_pred_smote = rf_tun_final.predict(X_test)

report_rf_smote = classification_report(y_test,
                                  rf_pred_smote,
                                  target_names=classe_map.values(),
                               
                                  output_dict=True)


print(classification_report(y_test, rf_pred_smote, target_names=classe_map.values()))
print("Random Forest - Relatório de Classificação com SMOTE:")
df_classificacao_smote = pd.DataFrame(report_rf_smote).transpose()

#%% Gráficos para avaliação de sobreajuste
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot


train_scores, test_scores = list(), list()

values = [i for i in range(1,21)]

# Avaliar com base na profundidade

for i in values:
    model = RandomForestClassifier(max_depth=i,
                                   min_samples_leaf=2,
                                   min_samples_split=2,
                                   n_estimators=100,
                                   max_features='sqrt',
                                   
                                   )
    model.fit(X_train, y_train)
    # Avaliar base de treino
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # Avaliar base de teste
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    
# Plotar gráfico de treino e teste scores vs profundidade

pyplot.plot(values, train_scores, '-o', label='Treino')
pyplot.plot(values, test_scores, '-o', label='Teste')
pyplot.legend()
# Eixos e formatação
plt.xlabel('Profundidade das árvores',fontsize=11, color='black')
plt.ylabel('Acurácia',fontsize=11, color='black')
plt.grid(False)
sns.despine(left=False, bottom=False)
    
ax = plt.gca()
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.5)
    
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.tight_layout()
pyplot.show()
    
#%% Validar robustez do modelo

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

cv_results = cross_validate(
    rf_tun_final, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'f1_weighted', 'roc_auc_ovo']
)
print("Acurácia média:", cv_results['test_accuracy'].mean())
print("F1-score médio:", cv_results['test_f1_weighted'].mean())



# Comparar desempenho em treino e teste

rf_tun_final.fit(X_train, y_train)
print("Acurácia (treino):", rf_tun_final.score(X_train, y_train))
print("Acurácia (teste):", rf_tun_final.score(X_test, y_test))


y_pred = rf_tun_final.predict(X_test)
print("F1 ponderado (teste):", f1_score(y_test, y_pred, average='weighted'))

y_proba = rf_tun_final.predict_proba(X_test)
print("ROC-AUC (teste):", roc_auc_score(y_test, y_proba, multi_class='ovo'))

#%% Matriz de confusão Random Forest

# Matriz de confusão RF
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_pred_smote), annot=True, fmt='d', cmap='viridis',
            xticklabels=classe_map.values(), yticklabels=classe_map.values())
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

#%% Curva ROC e AUC - Random Forest

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

classe_map = {0: 'Floresta', 1: 'Água', 2: 'Garimpo', 3: 'Vegetação_rasa'}
df['classe_nome'] = df['class'].map(classe_map)

# Binarizar os rótulos (para multiclasses ROC)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

rf_proba = rf_tun_final.predict_proba(X_test)

# Gerar curvas ROC para cada classe
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_proba[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

fpr_rf["micro"], tpr_rf["micro"], _ = roc_curve(y_test_bin.ravel(), rf_proba.ravel())
roc_auc_rf["micro"] = auc(fpr_rf["micro"], tpr_rf["micro"])

roc_auc_rf["macro"] = sum(roc_auc_rf[i] for i in range(n_classes)) / n_classes

plt.figure(figsize=(8, 6))
colors = ['green', 'blue', 'red', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2,
             label=f'{classe_map[i]} (AUC = {roc_auc_rf[i]:.2f})')

plt.plot(fpr_rf["micro"], tpr_rf["micro"], color='black', linestyle='--',
         label=f'Micro-average AUC = {roc_auc_rf["micro"]:.2f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('')
plt.legend(loc="lower right")
mpl.rcParams['font.family'] = 'DejaVu Sans'
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.tight_layout()
plt.grid(False)
sns.despine(left=False, bottom=False)
plt.show()


print("AUC por classe:")
for i in range(n_classes):
    print(f" - {classe_map[i]}: {roc_auc_rf[i]:.3f}")
print(f"\nAUC Micro: {roc_auc_rf['micro']:.3f}")
print(f"AUC Macro: {roc_auc_rf['macro']:.3f}")

#%% Feature Importance por Permutação no modelo Random Forest

def plot_grafico_norma(importancias, xlabel="Importância", ylabel="Variáveis"):
    # Ordenar crescentemente
    importancias = importancias.sort_values(ascending=True)
    # Paleta viridis com mesma ordem
    palette = sns.color_palette("viridis", len(importancias))
    # Fonte
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    # Criar figura
    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=importancias.values,
        y=importancias.index,
        palette=palette
    )
    # Eixos e formatação
    plt.xlabel(xlabel, fontsize=11, color='black')
    plt.ylabel(ylabel, fontsize=11, color='black')
    plt.grid(False)
    sns.despine(left=False, bottom=False)
    
    ax = plt.gca()
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    plt.tight_layout()
    plt.show()

# Calcular importância por permutação
perm_rf = permutation_importance(rf_tun_final, X_test, y_test, n_repeats=30, random_state=42)

# Garantir que 'indices' é a lista de colunas
indices = X_test.columns
importancias_rf = pd.Series(perm_rf.importances_mean, index=indices)

# Plot
plot_grafico_norma(importancias_rf, xlabel="Importância média por permutação no Random Forest", ylabel="Variáveis")

# 🔹 Importância por classe
importances_by_class = []

for target_class in sorted(y.unique()):
    class_name = df[df['class'] == target_class]['label'].iloc[0]
    y_binary = (y_test == target_class).astype(int)

    model_bin = rf_tun_final.fit(X_train, (y_train == target_class).astype(int))

    perm_bin = permutation_importance(model_bin, X_test, y_binary, n_repeats=30, random_state=42)
    importances_bin = pd.Series(perm_bin.importances_mean, index=indices)

    # Armazenar resultados
    importances_by_class.append({'class': target_class, 'class_name': class_name, 'importances': importances_bin})

    # Plot com norma
    plot_grafico_norma(importances_bin, xlabel=f"Importância - Classe: {class_name}")

# 🔹 Combinar resultados em DataFrame
data_for_df = {}
for item in importances_by_class:
    data_for_df[item['class_name']] = item['importances']

df_importancias = pd.DataFrame(data_for_df).T.round(4)

# 🔹 Exibir e salvar
print("\n🔹 Importância dos Índices por Classe:")
print(df_importancias)


#%% Features Importances

# Extrair importância das features por Mean Decrease in Impurity (MDI)
importances = rf_tun_final.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]  # Ordem decrescente


# Print das importâncias
print("Importância das Features:")
for i, idx in enumerate(indices):
    print(f"{features[idx]}: {importances[idx]:.4f}")
  
#%% Exportando modelo Random Forest
import joblib
import os

modelo_trein_rf = 'modelo_rf.joblib'
joblib.dump(rf_tun_final,modelo_trein_rf)
print(f'Modelo salvo com sucesso como {modelo_trein_rf}')

caminho = os.getcwd()
print(f'O código está sendo executado em {caminho}')

#%% Exportando as colunas de treino
import joblib

colunas_treino = X_train.columns.tolist()
joblib.dump(colunas_treino, 'colunas_treino.joblib')


#%% Modelo KNearest-Neighbors

#Definindo valor de k com validação cruzada cv=10
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
# ('scaler', StandardScaler()), caso precise normalizar os dados


k_range = range(1,21)
k_scores = []
k_error = []

for k in k_range:
    pipeline = Pipeline([
       ('smote', SMOTE(sampling_strategy='minority', 
                       random_state=42, k_neighbors=5)), 
       ('knn', KNeighborsClassifier(n_neighbors=k))
   ])
    
    scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')  
    k_scores.append(scores.mean())
    k_error.append(1 - scores.mean())

best_k = k_range[np.argmax(k_scores)]
print('Melhor k = ', best_k)
    
# Plotar resultados para accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_range, k_scores, marker='o', linestyle='-', color='navy')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia média (validação cruzada)')
plt.title('Acurácia x k (KNN)')
plt.xticks(k_range)
plt.grid(False)
ax = plt.gca()
sns.despine(left=False, bottom=False)

ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.tight_layout()
plt.show()


# Plotar resultados para error rate
plt.figure(figsize=(8, 5))
plt.plot(k_range, k_error, marker='o', linestyle='-', color='navy')
plt.xlabel('Valor de k')
plt.ylabel('Taxa de erro (validação cruzada)')
plt.title('Taxa de erro x k (KNN)')
plt.xticks(k_range)
plt.grid(False)

ax = plt.gca()
sns.despine(left=False, bottom=False)

ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.tight_layout()
plt.show()

#%% Modelo tunado KNN com SMOTE

# Treinar o modelo final com o melhor k
knn_tun_final = Pipeline([
    ('smote', SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

knn_tun_final.fit(X_train, y_train)
knn_pred = knn_tun_final.predict(X_test)


print(classification_report(y_test, knn_pred, target_names=classe_map.values()))

# Matriz de confusão KNN
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d', cmap='viridis',
            xticklabels=classe_map.values(), yticklabels=classe_map.values())
plt.title("Matriz de Confusão - KNN")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

#%% Robustez do modelo com SMOTE
cv_results = cross_validate(
    knn_tun_final, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'f1_weighted', 'roc_auc_ovo']
)
print("Acurácia média:", cv_results['test_accuracy'].mean())
print("F1-score médio:", cv_results['test_f1_weighted'].mean())


# Comparar desempenho em treino e teste

knn_tun_final.fit(X_train, y_train)
print("Acurácia (treino):", knn_tun_final.score(X_train, y_train))
print("Acurácia (teste):", knn_tun_final.score(X_test, y_test))
y_pred = knn_tun_final.predict(X_test)
print("F1 ponderado (teste):", f1_score(y_test, y_pred, average='weighted'))
y_proba = knn_tun_final.predict_proba(X_test)
print("ROC-AUC (teste):", roc_auc_score(y_test, y_proba, multi_class='ovo'))


#%% Curva ROC-AUC para modelo KNN
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Pipeline com SMOTE + scaler + KNN
knn_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

# Usar a pipeline dentro do OneVsRestClassifier
knn_final = OneVsRestClassifier(knn_pipeline)
knn_final.fit(X_train, y_train)

# Previsão de probabilidades com KNN
y_score_knn = knn_final.predict_proba(X_test)

# Gerar curvas ROC para cada classe
fpr_knn = dict()
tpr_knn = dict()
roc_auc_knn = dict()
for i in range(n_classes):
    fpr_knn[i], tpr_knn[i], _ = roc_curve(y_test_bin[:, i], y_score_knn[:, i])
    roc_auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])

# Calcular AUC macro e micro
fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(y_test_bin.ravel(), y_score_knn.ravel())
roc_auc_knn["micro"] = auc(fpr_knn["micro"], tpr_knn["micro"])
roc_auc_knn["macro"] = sum(roc_auc_knn[i] for i in range(n_classes)) / n_classes

# Plotar ROC para KNN
plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_knn[i], tpr_knn[i], color=color, lw=2,
             label=f'{classe_map[i]} (AUC = {roc_auc_knn[i]:.2f})')

plt.plot(fpr_knn["micro"], tpr_knn["micro"], color='black', linestyle='--',
         label=f'Micro-average AUC = {roc_auc_knn["micro"]:.2f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('')
plt.legend(loc="lower right")
mpl.rcParams['font.family'] = 'DejaVu Sans'
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.tight_layout()
plt.grid(False)
sns.despine(left=False, bottom=False)
plt.show()

# Exibir AUCs
print("AUC por classe - KNN:")
for i in range(n_classes):
    print(f" - {classe_map[i]}: {roc_auc_knn[i]:.3f}")
print(f"\nAUC Micro - KNN: {roc_auc_knn['micro']:.3f}")
print(f"AUC Macro - KNN: {roc_auc_knn['macro']:.3f}")

#%% Feature Importance com permutação para o modelo KNN


# Função de gráfico com formatação conforme norma

def plot_grafico_norma(importancias, xlabel="Importância", ylabel="Variáveis"):
    # Ordenar crescentemente
    importancias = importancias.sort_values(ascending=True)
    # Paleta viridis com mesma ordem
    palette = sns.color_palette("viridis", len(importancias))
    # Fonte
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    # Criar figura
    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=importancias.values,
        y=importancias.index,
        palette=palette
    )
    # Eixos e formatação
    plt.xlabel(xlabel, fontsize=11, color='black')
    plt.ylabel(ylabel, fontsize=11, color='black')
    plt.grid(False)
    sns.despine(left=False, bottom=False)
    
    ax = plt.gca()
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    plt.tight_layout()
    plt.show()

# Calcular importância por permutação
perm = permutation_importance(knn_final, X_test, y_test, n_repeats=30, random_state=42)

# Garantir que 'indices' é a lista de colunas
indices = X_test.columns
importancias = pd.Series(perm.importances_mean, index=indices)

# Plot
plot_grafico_norma(importancias, xlabel="Importância média por permutação no KNN", ylabel="Variáveis")

# 🔹 Importância por classe
importances_by_class = []

for target_class in sorted(y.unique()):
    class_name = df[df['class'] == target_class]['label'].iloc[0]
    y_binary = (y_test == target_class).astype(int)

    model_bin = knn_final.fit(X_train, (y_train == target_class).astype(int))

    perm_bin = permutation_importance(model_bin, X_test, y_binary, n_repeats=30, random_state=42)
    importances_bin = pd.Series(perm_bin.importances_mean, index=indices)

    # Armazenar resultados
    importances_by_class.append({'class': target_class, 'class_name': class_name, 'importances': importances_bin})

    # Plot com norma
    plot_grafico_norma(importances_bin, xlabel=f"Importância - Classe: {class_name}")

# 🔹 Combinar resultados em DataFrame
data_for_df = {}
for item in importances_by_class:
    data_for_df[item['class_name']] = item['importances']

df_importancias = pd.DataFrame(data_for_df).T.round(4)

# 🔹 Exibir e salvar
print("\n🔹 Importância dos Índices por Classe:")
print(df_importancias)


#%% Comparação entre os modelos - Random Forest e KNN

# 5x2cv paired t test

from mlxtend.evaluate import paired_ttest_5x2cv


t, p = paired_ttest_5x2cv(estimator1=rf_tun_final,
                          estimator2=knn_tun_final,
                          X=X, y=y,
                          scoring='accuracy',
                          random_seed=1)

print('t statistic: %.3f' % t)
print('p-value corrigido: %.3f' % p)

