import pandas as pd
from sklearn.preprocessing import LabelEncoder

tabela = pd.read_csv("clientes.csv")
print(tabela)

print(tabela.info())
print(tabela.columns)


from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()


for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])


print(tabela.info())


x = tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]

from sklearn.model_selection import train_test_split


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvore = RandomForestClassifier() 
modelo_knn = KNeighborsClassifier() 


modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)


contagem_scores = tabela["score_credito"].value_counts()
print(contagem_scores['Standard'] / sum(contagem_scores))


from sklearn.metrics import accuracy_score


previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())


print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))


novos_clientes = pd.read_csv("novos_clientes.csv")
print(novos_clientes)
for coluna in novos_clientes.columns:
    if novos_clientes[coluna].dtype == "object" and coluna != "score_credito":
        novos_clientes[coluna] = codificador.fit_transform(novos_clientes[coluna])

previsoes = modelo_arvore.predict(novos_clientes)
print(previsoes)



colunas = list(x_teste.columns)
importancia = pd.DataFrame(index=colunas, data=modelo_arvore.feature_importances_)
importancia = importancia * 100
print(importancia)