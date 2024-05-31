var compositeToUse = ee.Image("projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec"),
    exampleFC = ee.FeatureCollection("users/laubert/DominanceAndRarity/Scaled_Bootstrapped_Data/ScaledDominanceAndRarity_BootstrapColl_001");

// !! Input the name of the property being modelled
// Decide between 
// - Dominance
// - Dominance_scaled
// - Rarity
// - Rarity_scaled
var propToModel = 'Rarity_scaled';

// !! Change the start and end number of the bootstrapped models 
var start = 400;
var end =   500;

// Print info on the points and display them (optionally)
print('Example Training FC',exampleFC.limit(5));
print('Size of Training FC',exampleFC.size());
Map.addLayer(exampleFC);

// Make a list of covariates to use
var covarsToUse = exampleFC.first().propertyNames().removeAll([
	'system:index',
	'Pixel_Long',
	'Pixel_Lat',
	'Resolve_Biome',
	'Dominance',
	'Dominance_scaled',
	'Rarity',
	'Rarity_scaled'
]);
print('Covariates being used', covarsToUse.sort());

// Prepare the composite for use
var preparedComposite = compositeToUse.select(covarsToUse);

// Instantiate a classifier of interest
var randomForestClassifier = ee.Classifier.randomForest({
	numberOfTrees: 100,
	variablesPerSplit: 3,
	bagFraction: 0.632,
	seed: 1
}).setOutputMode('REGRESSION');

// Input a base image name for exporting and organizational purposes
var bootstrapFileName = 'ScaledDominanceAndRarity_BootstrapColl_';

// Input the home path of the training collections
var homeCollectionPath = 'users/laubert/DominanceAndRarity/Scaled_Bootstrapped_Data';

// Make a list of seeds to use for the bootstrapping
function JSsequence(i) {return i ? JSsequence(i - 1).concat(i) : []}
var numberOfSubsets = 500;
var seedsForBootstrappingPrep = JSsequence(numberOfSubsets);
var seedsForBootstrapping = seedsForBootstrappingPrep.slice(start,end);
print(seedsForBootstrappingPrep);
print(seedsForBootstrapping);

// Create an unbounded geometry for exports
var unboundedGeo = ee.Geometry.Polygon([-180, 88, 0, 88, 180, 88, 180, -88, 0, -88, -180, -88], null, false);

// Load the bootstrap function
var bootStrap = require('users/devinrouth/toolbox:Stratified_Bootstrap_FeatureCollection.js');

// Make a function to pad numbers with leading zeroes for formatting purposes
function pad(num, size) {
    var s = num+"";
    while (s.length < size) s = "0" + s;
    return s;
}

// Apply the machine learning using all collections
var imageCollectionToMap = ee.ImageCollection(seedsForBootstrapping.map(function(seedToUse) {
 return ee.Image(0).set('TrainingColl',ee.FeatureCollection(homeCollectionPath+'/'+bootstrapFileName+pad(seedToUse,3)));
}));

var imageCollectionToReduce =  imageCollectionToMap.map(function(i) {

	// Load the feature collection with the training data
	var trainingColl = ee.FeatureCollection(i.get('TrainingColl'));

	// Train the classifers with the sampled points
	var trainedBootstrapClassifier = randomForestClassifier.train({
		features: trainingColl,
		classProperty: propToModel,
		inputProperties: covarsToUse
	});

	// Apply the classifier to the composite to make the final map
	var bootstrapImage = preparedComposite.classify(trainedBootstrapClassifier).rename(propToModel);
	
	return bootstrapImage;

});


// Once the boostrap iterations are complete, calculate the mean and the standard deviation
print('Image Collection of Bootstrapped Images',imageCollectionToReduce);

var meanStdDevImage = imageCollectionToReduce.reduce({
  reducer:ee.Reducer.mean().combine({reducer2:ee.Reducer.stdDev(),sharedInputs:true})
});
// print('Final Bootstrapped Image',meanStdDevImage);

// Export the image
Export.image.toAsset({
	image: meanStdDevImage,
	description: propToModel + '_Bootstrap_Mean_StdDev_' + start + 'to' + end,
	assetId: 'users/laubert/DominanceAndRarity/Rarity_scaled_new/' + propToModel + '_Bootstrap_Mean_StdDev_' + start + 'to' + end,
	region: unboundedGeo,
	crs: 'EPSG:4326',
	crsTransform: [0.008333333333333333, 0, -180, 0, -0.008333333333333333, 90],
	maxPixels: 1e13
});

