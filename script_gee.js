var area = ee.Geometry.Polygon(
        [[[-63.30274212610604, 2.4294737010294583],
          [-63.30274212610604, 2.257269873105874],
          [-62.84337628137948, 2.257269873105874],
          [-62.84337628137948, 2.4294737010294583]]], null, false);



// Centralizar o mapa na área de estudo
Map.centerObject(area, 12); // o segundo argumento é o nível de zoom

//Calcular a área da região de estudo
var area_m2  = area.area({maxError: 1}); 
var area_km2 = area_m2.divide(1e6); 
print('Área (km²):', area_km2);

//------------------------Seleção da coleção----------------------------------------------------------

//Aplicando uma máscara de núvens na coleção Sentinel-2/
 
function maskS2clouds(image) {
  
  var qaBand = image.select('QA60');
  
  // Bits 10 e 11 são nuvens e sombra de nuvens
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  // Máscara para condições claras
  var mask = qaBand.bitwiseAnd(cloudBitMask).eq(0)
      .and(qaBand.bitwiseAnd(cirrusBitMask).eq(0));
  
  // Fatores de escala para Sentinel-2 (reflectância multiplicada por 10000)
  var opticalBands = image.select('B.*').divide(10000);
  
  return image
      .updateMask(mask)
      .addBands(opticalBands, null, true)
      .clip(area)
      .copyProperties(image, image.propertyNames())
      .set({date: image.date().format('YYYY-MM-dd')});
}


//---------------------------------------Função dos índices-------------------------------------------
 
function indices (image) {
  //Indice de Vegetação 
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');// Rouse 1973

  //Diferenciar vegetação de solo exposto e áreas degradadas
  // SAVI (L = 0.5)
  var savi = image.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('SAVI');
   
        
    //Índices de Água
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI'); //Mc Feeters 1996
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI'); // Xu 2006
   
   // Índices turbidez
  var ndti = image.normalizedDifference(['B4', 'B3']).rename('NDTI');

  //Diferenciar áreas alagadas de garimpo
  var ndre = image.normalizedDifference(['B8A', 'B5']).rename('NDRE');


    
  return image.addBands([ndvi,mndwi,ndti,ndre,savi])}
  
//-------------------------/Importando coleção Sentinel-2/-------------------------------------------

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(area) // Filtro pela reserva
    .filterDate('2024-01-01', '2024-12-31') // Período ampliado
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) // Limite de nuvens mais restrito
    .map(maskS2clouds)
    .map(indices);

print('Coleção Sentinel-2 filtrada:', s2);
print ('Quantidade de imagens:', s2.size());

// Criar imagem mediana
var s2_reduce = s2.median();
var s2_reduce_bands = s2_reduce.select(['NDVI','MNDWI','NDTI','NDRE','SAVI','B.*']);//'AWEI_sh','NDRE','PSRI','MNDWI_G','SMI'
print('Bandas disponíveis:', s2_reduce_bands.bandNames());
var bands = s2_reduce_bands.bandNames();

// Visualização
Map.addLayer(s2_reduce_bands, {bands: ['B4','B3','B2'], min: 0, max: 0.3}, 'Sentinel-2 RGB');

//-------------------------/Criação de amostras/-----------------------------------------------------

var amostrasAgua = s2_reduce_bands.sampleRegions({
  collection: Agua,
  scale: 10,
  geometries: true
}).randomColumn('random');

print('Número de amostras de Agua', amostrasAgua);

var amostrasFloresta = s2_reduce_bands.sampleRegions({
  collection: Floresta,
  scale: 10,
  geometries: true
}).randomColumn('random');

print('Número de amostras de Floresta', amostrasFloresta);

var amostrasGarimpo = s2_reduce_bands.sampleRegions({
  collection: Garimpo,
  scale: 10,
  geometries: true
}).randomColumn('random');

print('Número de amostras de Garimpo', amostrasGarimpo);

var amostrasVegetacao_rasa = s2_reduce_bands.sampleRegions({
  collection: Vegetacao_rasa,
  scale: 10,
  geometries: true
}).randomColumn('random');

print('Número de amostras de Vegetacao_rasa', amostrasVegetacao_rasa);


//----------------------------------------------------------------------------------------
//  Juntando as amostras em uma única feature
var labels = amostrasFloresta.merge(amostrasAgua)
                             .merge(amostrasGarimpo)
                             .merge(amostrasVegetacao_rasa);
                             //.merge(amostrasSolo_exposto);

// // Criaçao de layers das amostras de cada classe
Map.addLayer(amostrasAgua, {color: '0000FF '}, 'Água (validação)');
Map.addLayer(amostrasGarimpo, {color: 'af2a2a '}, 'Garimpo (validação)');
Map.addLayer(amostrasFloresta, {color: '006400 '}, 'Floresta (validação)');
Map.addLayer(amostrasVegetacao_rasa, {color: 'B8AF4F '}, 'Vegetacao_rasa (validação)');

//------------------------------------EXPORTAÇÃO DOS DADOS----------------------------------

Export.table.toDrive({
  collection: labels,
  description: 'Exporta_Amostras_Indices',
  folder: 'GEE_exports', 
  fileNamePrefix: 'amostras_indices_espectrais',
  fileFormat: 'CSV'
});

//-----------------------MAPAS DE CLASSIFICAÇÃO APÓS AJUSTE DOS PARÂMETROS-------------------

var n = 200;
var balancedTraining = ee.FeatureCollection([
  amostrasFloresta.randomColumn().limit(n),
  amostrasAgua.randomColumn().limit(n), 
  amostrasGarimpo.randomColumn().limit(n),
  amostrasVegetacao_rasa.randomColumn().limit(n)
]).flatten();

// 1. Separar dados ANTES do treinamento
var split = labels.randomColumn('split');
var training = split.filter(ee.Filter.lt('split', 0.7));
var testing = split.filter(ee.Filter.gte('split', 0.7));

//------------------------------------Random Forest------------------------------------------
// 2. Treinar com dados balanceados
var classifier = ee.Classifier.smileRandomForest({
  //Hiperparâmetros após Grid Search CV  
  numberOfTrees: 100,  
  variablesPerSplit: null, // Padrão: sqrt(n_features)
  minLeafPopulation: 2, 
  bagFraction: 0.7,
  maxNodes: null,
  seed: 123
}).train({
  features: balancedTraining,
  classProperty: 'class', 
  inputProperties: bands
});

var classified = s2_reduce_bands.classify(classifier);

// Exibir as entradas e os resultados.
Map.addLayer(classified,
            {min: 0, max: 3, palette: ['006400', '0000FF', 'af2a2a', 'B8AF4F']},
            'Classificação RF');

//---------------------------------KNN--------------------------------------------
// Configurar o classificador KNN
var knnClassifier = ee.Classifier.smileKNN(3,"Auto","EUCLIDEAN").train({
  features: balancedTraining,  // Dados de treinamento
  classProperty: 'class',  // Propriedade de classe
  inputProperties: bands  // Bandas utilizadas
});

// Classificar a imagem usando KNN
var classifiedKNN = s2_reduce_bands.classify(knnClassifier);

// Adicionar a classificação KNN ao mapa
Map.addLayer(classifiedKNN,
            {min: 0, max: 3, palette: ['006400', '0000FF', 'af2a2a', 'B8AF4F']},
            'Classificação KNN');

