# Classificação de Garimpo na Amazônia
O objetivo desse projeto é comparar o desempenho dos algoritmos Random Forest e KNN na classificação de garimpos na Amazônia. O trabalho foi desenvolvido como projeto de TCC para obtenção do título de Especialista em Data Science e Analytics pela USP Esalq.

Resumo executivo na Revista E&S:
Link: https://revistaes.com.br/resumo-executivo/identificacao-de-garimpo-na-amazonia-com-aprendizagem-de-maquina-e-indices-espectrais

# Fluxograma com as etapas do projeto
<img width="1468" height="562" alt="Image" src="https://github.com/user-attachments/assets/4a3920c8-1b03-4a2c-9ed0-452c21cd2e44" />


Resumo: 

A Floresta Amazônica é um importante bioma do Brasil e do mundo, lar de diversos povos originários; porém nas últimas décadas diversas reservas indígenas tornaram-se alvo da exploração ilegal de garimpeiros. 
Diante desse cenário, a adoção de novas tecnologias para monitoramento dessas terras é de suma importância. O sensoriamento remoto oferece uma solução prática de monitoramento, visto que essas regiões podem oferecer difícil acesso e os dados de satélite são públicos e gratuitos. 
Este estudo tem por objetivo identificar e monitorar áreas de garimpo ilegal na Floresta Amazônica, dentro da reserva Yanomami, por meio da aplicação de algoritmos de classificação supervisionada, Random Forest [RF] e K-Nearest Neighbors [K-NN], utilizando os índices espectrais de vegetação (NDVI, SAVI, NDRE) e índices de água (MNDWI, NDTI) como variáveis do modelo. 
As imagens utilizadas para extração dos índices foram obtidas da coleção do satélite Sentinel-2, no ano de 2024. 
O RF apresentou acurácia média de 95% no conjunto teste e AUC médio de 1.00 para a classe “Garimpo”, enquanto o K-NN apresentou acurácia média de 93% no teste e AUC de 0.98 para “Garimpo”. 
Os modelos apresentaram resultados estatisticamente semelhantes, segundo o teste 5x2cv. 
O índice de maior importância para os modelos foi o NDTI, seguido dos índices NDRE e SAVI. Dessa forma, os algoritmos de classificação apresentam aplicabilidade no monitoramento de alertas dentro das reservas na Amazônia, além de oferecer suporte tecnológico para os órgãos responsáveis.
