// Get the GFBI dataset
var GFBI = ee.FeatureCollection("users/laubert/DominanceAndRarity/ScaledDominanceRarityData");

// Visualize the dataset 
// var dominance = ee.Image().float().paint(GFBI, 'Dominance').focal_mean(2);
var dominance_scaled = ee.Image().float().paint(GFBI, 'Dominance_scaled').focal_mean(2);
// var rarity = ee.Image().float().paint(GFBI, 'Rarity').focal_mean(2);
var rarity_scaled = ee.Image().float().paint(GFBI, 'Rarity_scaled').focal_mean(2);

// Map.addLayer(dominance, {min: 10, max: 100, palette: ['FFFFFF','#003700']}, 'Dominance');
Map.addLayer(dominance_scaled, {min: -2, max: 3.1, palette: ['FFFFFF','#003700']}, 'Dominance scaled');
// Map.addLayer(rarity, {min: 0, max: 100, palette: ['FFFFFF','#003700']}, 'Rarity');
Map.addLayer(rarity_scaled, {min: -6, max: 2.2, palette: ['FFFFFF','#003700']}, 'Rarity scaled');

// Load the composite holding all covariate layers
var composite = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec');

// Select the layers of interest
var composite_selected = composite.select([
  'CHELSA_BIO_Annual_Mean_Temperature', 
  'CHELSA_BIO_Annual_Precipitation',
  'CHELSA_BIO_Isothermality',
  'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
  'CHELSA_BIO_Min_Temperature_of_Coldest_Month',
  'CHELSA_BIO_Precipitation_Seasonality',
  'CHELSA_BIO_Precipitation_of_Driest_Month',
  'CHELSA_BIO_Precipitation_of_Wettest_Month',
  'CHELSA_BIO_Temperature_Annual_Range',
  'CHELSA_BIO_Temperature_Seasonality',
  'MODIS_EVI', 
  'EarthEnvTexture_Dissimilarity_EVI',
  'EarthEnvTexture_Shannon_Index',
  'EarthEnvTopoMed_Elevation',
  'EarthEnvTopoMed_Slope',
  'GlobBiomass_AboveGroundBiomass',
  'GlobBiomass_GrowingStockVolume',
  'ConsensusLandCover_Human_Development_Percentage',
  'MODIS_LAI',
  'MODIS_NDVI',
  'MODIS_Nadir_Reflectance_Band1',
  'MODIS_Nadir_Reflectance_Band2',
  'MODIS_Nadir_Reflectance_Band3',
  'MODIS_Nadir_Reflectance_Band4',
  'MODIS_Nadir_Reflectance_Band5',
  'MODIS_Nadir_Reflectance_Band6',
  'MODIS_Nadir_Reflectance_Band7',
  'MODIS_NPP', 
  'Pixel_Lat',
  'Pixel_Long',
  'GPWv4_Population_Density', 
  'SG_CEC_015cm', 
  'SG_Clay_Content_015cm',
  'SG_SOC_Density_015cm',
  'SG_Sand_Content_015cm', 
  'SG_Saturated_H2O_Content_015cm',
  'SG_Silt_Content_015cm', 
  'SG_Soil_pH_H2O_015cm', 
  'CrowtherLab_Tree_Density',
  'GFAD_FractionOfRegrowthForest_downsampled50km',
  'GFAD_regrowthForestAge_Mean_downsampled50km',
  'Resolve_Biome',
  'NASA_ForestCanopyHeight'
  ]);

// Sample the composite using at the GFBI locations
var regressionMatrix = GFBI.map(function(f){
  return f.set(composite_selected.reduceRegion('first',f.geometry()));
});

// Export the Regression Matrix to Google Drive 
Export.table.toDrive({
  collection: regressionMatrix, 
  description: '20200901_RegressionMatrix_scaledDominanceAndRarity',
  folder: 'GEE_Output'
});
