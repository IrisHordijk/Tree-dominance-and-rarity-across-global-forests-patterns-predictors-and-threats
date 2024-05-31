
// Decide between 
// - Dominance
// - Dominance_scaled
// - Rarity
// - Rarity_scaled
var propToModel = 'Dominance_scaled';

// Load the ImageCollections
var imageToAssess = ee.ImageCollection([
  ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_0to100'),
  ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_100to200'),
  ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_200to300'),
  ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_300to400'),
  ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_400to500'),
  ]).mean()
    .reproject(ee.Image('users/laubert/DominanceAndRarity/'+propToModel+'_new/'+propToModel+'_Bootstrap_Mean_StdDev_0to100').projection())
    .select(0);
print(imageToAssess, 'Image to assess');
// Load the GFBI training points 
var trainingPoints = ee.FeatureCollection('users/laubert/DominanceAndRarity/DominanceRarityData')
print(trainingPoints.size())
var trainingPoints = trainingPoints.limit(2000);
print(trainingPoints)


//------------------------------------------------------------------------------
//--- SAMPLE PREDICTED VS OBSERVED ---------------------------------------------
//------------------------------------------------------------------------------
// Compute an R-squared from the training points and the final image

// Sample the filled images with the original points
var predsampledPoints = imageToAssess.sampleRegions({collection:trainingPoints.select(propToModel), geometries:true});

// Make a function to compute the R-squared
var computeCoeffOfDetermination = function(inputtedFC, mainPropName, predictedPropName) {

	// Compute the average value of the main property of interest
	var mainValueMean = ee.Number(ee.Dictionary(inputtedFC.reduceColumns('mean', [mainPropName])).get('mean'));

	// Compute the total sum of squares
	var totalSumOfSquaresFC = inputtedFC.map(function(f) {
		return f.set('DependVarMeanDiffs', ee.Number(f.get(mainPropName)).subtract(mainValueMean).pow(2));
	});
	var totalSumOfSquares = ee.Number(totalSumOfSquaresFC.reduceColumns('sum', ['DependVarMeanDiffs']).get('sum'));

	// Compute the residual sum of squares
	var residualSumOfSquaresFC = inputtedFC.map(function(f) {
		return f.set('Residuals', ee.Number(f.get(mainPropName)).subtract(ee.Number(f.get(predictedPropName))).pow(2));
	});
	var residualSumOfSquares = ee.Number(residualSumOfSquaresFC.reduceColumns('sum', ['Residuals']).get('sum'));

	// Finalize the calculation
	var finalR2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares));

	return finalR2;
};
print('Final R^2',computeCoeffOfDetermination(predsampledPoints,propToModel,propToModel+'_mean'));
print(predsampledPoints);

// Compute a scatter chart 
var sampledPointsChart = ui.Chart.feature.byFeature(predsampledPoints.map(function(f){return f.set('unit',f.get(propToModel))}), propToModel)
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Predicted vs. Observed',
    hAxis: {title: 'Observed'},
    vAxis: {title: 'Predicted'},
    pointSize: 5,
    trendlines: { 1: {} },
    series: {
        0: { pointSize: 5},
        1: { pointSize: 0}
    },
    legend:{ position: 'none' }
  });
print(sampledPointsChart);

// Export the sampled points to Drive 
Export.table.toDrive({
	collection: predsampledPoints,
	description: '4_PredictedMap_SampleValues_' + propToModel + '_toDrive',
	fileNamePrefix: 'Predicted_vs_Observed_' + propToModel,
	folder: 'GEE_Output'
});
print('!Run 4_PredictedMap_SampleValues_[PFT]_toDrive');


