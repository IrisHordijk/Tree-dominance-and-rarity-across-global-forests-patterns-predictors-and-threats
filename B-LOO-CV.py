# Import the modules of interest
import pandas as pd
import ee
from sklearn.metrics import r2_score
ee.Initialize()

varToModel = 'Dominance_scaled'
finalAssetID = 'users/laubert/DominanceAndRarity/bloo_cv_ScaledDominance_463'
fc_sampled = ee.FeatureCollection("users/laubert/DominanceAndRarity/Scaled_Bootstrapped_Data/ScaledDominanceAndRarity_BootstrapColl_463")
#gfbi = ee.FeatureCollection("users/laubert/DominanceAndRarity/ScaledDominanceRarityData")
#sampled = composite.sampleRegions(dataset, None, 927.6624232772797, geometries = True)

compositeImg = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec');

covarsToUse = fc_sampled.first().propertyNames().removeAll([
	'system:index',
	'Pixel_Long',
	'Pixel_Lat',
	'Resolve_Biome',
	'Dominance',
	'Dominance_scaled',
	'Rarity',
	'Rarity_scaled'
])

composite = compositeImg.select(covarsToUse)

# Define buffer sizes to test
buffer_sizes = [1000, 5000, 10000, 25000, 50000, 100000]

# Blocked Leave One Out cross-validation function:
def BLOOcv(f):
    # Test feature
    testFC = ee.FeatureCollection(f)

    # Training set: all samples not within geometry of test feature
    trainFC = fc_sampled.filter(ee.Filter.geometry(testFC).Not())

    # Classifier to test
    classifier = ee.Classifier.smileRandomForest(
                        numberOfTrees=100,
                        variablesPerSplit=3,
#                         minLeafPopulation=3,
                        bagFraction=0.632,
                        seed=1
                        ).setOutputMode('REGRESSION')

    # Train classifier
    trainedClassifer = classifier.train(trainFC, varToModel, covarsToUse)

    # Apply classifier
    classified = testFC.classify(classifier = trainedClassifer,
                                 outputName = 'predicted')

    # Get predicted value
    predicted = classified.first().get('predicted')

    return f.set('predicted', predicted).copyProperties(f)


# Define the R^2 function
def coefficientOfDetermination(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the mean of the property of interest
    propertyOfInterestMean = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).select([propertyOfInterest]).reduceColumns(ee.Reducer.mean(),[propertyOfInterest])).get('mean'));

    # Compute the total sum of squares
    def totalSoSFunction(f):
        return f.set('Difference_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(propertyOfInterestMean).pow(ee.Number(2)))
    totalSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).reduceColumns(ee.Reducer.sum(),['Difference_Squared'])).get('sum'))

    # Compute the residual sum of squares
    def residualSoSFunction(f):
        return f.set('Residual_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))
    residualSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).reduceColumns(ee.Reducer.sum(),['Residual_Squared'])).get('sum'))

    # Finalize the calculation
    r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))

    return ee.Number(r2)

bloo_cv_fc = ee.FeatureCollection(ee.List(buffer_sizes).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',n)))


# R2 calc function
def calc_final_r2(buffer_feat):
    # Add buffer to FC of sampled observations
    buffer = buffer_feat.get('buffer_size')

    fc_wBuffer = fc_sampled.map(lambda f: f.buffer(buffer))

    # Apply blocked leave one out CV function
    predicted = fc_wBuffer.map(BLOOcv)

    # Calculate R2 value
    R2_val = coefficientOfDetermination(predicted, varToModel, 'predicted')

    return(buffer_feat.set('R2_val', R2_val))

    ########################
    ## Uncomment the lines below to export the predicted/observed data per buffer size
    # predObs = predicted.select([varToModel, 'predicted'])
    # to_export = predObs.toList(50000).getInfo()
    # result = []
    # for item in to_export:
    #     values = item['properties']
    #     row = [str(values[key]) for key in [varToModel, 'predicted']]
    #     row = ",".join(row)
    #     result.append(row)
    #
    # df = pd.DataFrame([item.split(",") for item in result], columns = [varToModel, 'predicted'])
    # df['buffer_size'] = buffer
    # with open('temp/exported_df.csv', 'a') as f:
    #     df.to_csv(f, mode='a', header=f.tell()==0)


# Calculate R2 across range of R2 values
final_fc = bloo_cv_fc.map(calc_final_r2)

# Export FC to assets
bloo_cv_fc_export = ee.batch.Export.table.toAsset(
    collection = final_fc,
    description = varToModel+'bloo_cv',
    assetId = finalAssetID
);

bloo_cv_fc_export.start()

print('Blocked Leave-One-Out started!')
