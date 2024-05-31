# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import time
# import datetime
import ee
from functools import partial
# import os
# from functools import partial
# from pathlib import Path
# from scipy.spatial import ConvexHull
# from sklearn.decomposition import PCA
# from itertools import combinations
# from itertools import repeat
# from pathlib import Path

ee.Initialize()

# Define the project asset id
projectId = 'users/laubert/DominanceAndRarity/'

# Define geometries of interest
unboundedGeo = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)

# Define the covariates to use
covariateList = ee.FeatureCollection('users/laubert/DominanceAndRarity/Scaled_Bootstrapped_Data/ScaledDominanceAndRarity_BootstrapColl_001').first().propertyNames().removeAll([
	'system:index',
	'Pixel_Long',
	'Pixel_Lat',
	'Resolve_Biome',
	'Dominance',
	'Dominance_scaled',
	'Rarity',
	'Rarity_scaled'
]);

# Define the composite
compositeToClassify = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec').select(covariateList)

# Define the folder that holds the bootstrapped data
folderBootstrappedData = 'users/laubert/DominanceAndRarity/Scaled_Bootstrapped_Data/'

# Proportion of variance to be covered by the PCA for interpolation/extrapolation
propOfVariance = 90

####################################################################################################################################################################
# Configuration and project-specific settings
####################################################################################################################################################################
# Input the name of the username that serves as the home folder for asset storage
usernameFolderString = ''

# Input the Cloud Storage Bucket that will hold the bootstrap collections when uploading them to Earth Engine
# !! This bucket should be pre-created before running this script
bucketOfInterest = ''

# Specify file name of raw point collection (without extension); must be a csv file; don't include '.csv'
titleOfRawPointCollection = ''

# Input the name of the classification property
classProperty = ''

# Input the name of the project folder inside which all of the assets will be stored
# This folder will be generated automatically in GEE
projectFolder = ''

# Specify the column names where the latitude and longitude information is stored: these columns must be present in the csv containing the observations
latString = 'latitude'
longString = 'longitude'

# Name of a local folder holding input data
holdingFolder = ''

# Name of a local folder for output data
outputFolder = ''

# Create directory to hold training data
Path(outputFolder).mkdir(parents=True, exist_ok=True)

# Path to location of ee and gsutil python dependencies
bashFunction_EarthEngine = ''
bashFunctionGSUtil = ''

# Perform modeling in log space? (True or False)
log_transform_classProperty = False

# Ensemble of top 10 models from grid search? (True or False)
ensemble = True



####################################################################################################################################################################
# Export settings
####################################################################################################################################################################
# Set pyramidingPolicy for exporting purposes
pyramidingPolicy = 'mean'

# Specify CRS to use (of both raw csv and final maps)
CRStoUse = 'EPSG:4326'

# Geometry to use for export
exportingGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False);

# Set resolution of final image in arc seconds (30 arc seconds equals to ± 927m at the equator)
export_res = 30

# Convert resolution to degrees
res_deg = export_res/3600

####################################################################################################################################################################
# General settings
####################################################################################################################################################################

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

####################################################################################################################################################################
# Covariate data settings
####################################################################################################################################################################

# List of the covariates to use
covariateList = [
"CHELSA_BIO_Annual_Mean_Temperature",
"CHELSA_BIO_Annual_Precipitation",
"CHELSA_BIO_Precipitation_Seasonality",
"CHELSA_BIO_Temperature_Annual_Range",
"EarthEnvTopoMed_Elevation",
"EarthEnvTopoMed_Slope",
"EarthEnvTopoMed_TopoPositionIndex",
"GHS_Population_Density",
"HansenEtAl_TreeCover_Year2010",
"IPCC_Global_Biomass",
"MODIS_NDVI",
"SG_CEC_015cm",
"SG_Depth_to_bedrock",
"SG_SOC_Content_015cm",
"SG_Sand_Content_015cm",
"SG_Soil_pH_H2O_015cm"
]

# Load the composite on which to perform the mapping, and subselect the bands of interest
full_composite = ee.Image("users/crowtherlab/References/example_composite_30ArcSec")
compositeToClassify = ee.Image("users/crowtherlab/References/example_composite_30ArcSec").select(covariateList)

# Scale of composite
scale = full_composite.projection().nominalScale().getInfo()

####################################################################################################################################################################
# Additional settings
####################################################################################################################################################################

####################################################################################################################################################################
# RF and Cross validation settings
####################################################################################################################################################################
# Grid search parameters; specify range
# variables per split
varsPerSplit_list = list(range(2,8))

# minium leaf population
leafPop_list = [3,4,5]

# Set k for k-fold CV and make a list of the k-fold CV assignments to use
k = 10
kList = list(range(1,k+1))

# Metric to use for sorting k-fold CV hyperparameter tuning
sort_acc_prop = 'Mean_R2' # (either one of 'Mean_R2', 'Mean_MAE', 'Mean_RMSE')

# Set number of trees in RF models
nTrees = 250

# Spatial leave-one-out cross-validation settings
# skip test points outside training space after removing points in buffer zone? This might reduce extrapolation but overestimate accuracy
loo_cv_wPointRemoval = False

# Define buffer size in meters; use Moran's I or other test to determine SAC range
# Alternatively: specify buffer size as list, to test across multiple buffer sizes
buffer_size = 250000

# Input the name of the property that holds the CV fold assignment
cvFoldString = 'CV_Fold'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = classProperty+"_training_data"

assetIDForCVAssignedColl = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_training_data'

####################################################################################################################################################################
# Bootstrap settings
####################################################################################################################################################################
# Number of bootstrap iterations
bootstrapIterations = 100

# Generate the seeds for bootstrapping
seedsToUseForBootstrapping = list(range(1, bootstrapIterations+1))

# Input the name of a folder used to hold the bootstrap collections
bootstrapCollFolder = 'Bootstrap_Collections'

# Input the header text that will name each bootstrapped dataset
fileNameHeader = classProperty+'BootstrapColl_'

# Stratification inputs
# Write the name of the variable used for stratification
# !! This variable should be included in the input dataset
stratificationVariableString = "Resolve_Biome"

# Input the dictionary of values for each of the stratification category levels
# !! This area breakdown determines the proportion of each biome to include in every bootstrap
strataDict = {
	1: 14.900835665820974,
	2: 2.941697660221864,
	3: 0.526059731441294,
	4: 9.56387696566245,
	5: 2.865354077500338,
	6: 11.519674266872787,
	7: 16.26999434439293,
	8: 8.047078485979089,
	9: 0.861212221078014,
	10: 3.623974712557433,
	11: 6.063922959332467,
	12: 2.5132866428302836,
	13: 20.037841544639985,
	14: 0.26519072167008,
}

####################################################################################################################################################################
# Bash settings
####################################################################################################################################################################

# Specify main bash functions being used
bashFunction_EarthEngine = '/Users/Thomas/opt/anaconda3/envs/ee/bin/earthengine'

# Specify the arguments to these functions
assetIDStringPrefix = '--asset_id='
arglist_CreateCollection = ['create','collection']
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_List = ['ls']
arglist_Delete = ['rm','-r']
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_List = [bashFunction_EarthEngine]+arglist_List
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder


####################################################################################################################################################################
# Helper functions
####################################################################################################################################################################
# Function to convert FeatureCollection to Image
# def fcToImg(f):
# 	# Reduce to image, take mean per pixel
# 	img = sampledFC.reduceToImage(
# 		properties = [f],
# 		reducer = ee.Reducer.mean()
# 	)
# 	return img
#
# # Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
# def GEE_FC_to_pd(fc):
# 	result = []
# 	# Fetch data as a list
# 	values = fc.toList(100000).getInfo()
# 	# Fetch column names
# 	BANDS = fc.first().propertyNames().getInfo()
# 	# Remove system:index if present
# 	if 'system:index' in BANDS: BANDS.remove('system:index')
#
# 	# Convert to data frame
# 	for item in values:
# 		values = item['properties']
# 		row = [str(values[key]) for key in BANDS]
# 		row = ",".join(row)
# 		result.append(row)
#
# 	df = pd.DataFrame([item.split(",") for item in result], columns = BANDS)
# 	df.replace('None', np.nan, inplace = True)
#
# 	return df
#
# # R^2 function
# def coefficientOfDetermination(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
# 	# Compute the mean of the property of interest
# 	propertyOfInterestMean = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).select([propertyOfInterest]).reduceColumns(ee.Reducer.mean(),[propertyOfInterest])).get('mean'));
#
# 	# Compute the total sum of squares
# 	def totalSoSFunction(f):
# 		return f.set('Difference_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(propertyOfInterestMean).pow(ee.Number(2)))
# 	totalSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).reduceColumns(ee.Reducer.sum(),['Difference_Squared'])).get('sum'))
#
# 	# Compute the residual sum of squares
# 	def residualSoSFunction(f):
# 		return f.set('Residual_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))
# 	residualSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).reduceColumns(ee.Reducer.sum(),['Residual_Squared'])).get('sum'))
#
# 	# Finalize the calculation
# 	r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))
#
# 	return ee.Number(r2)
#
# # RMSE function
# def RMSE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
# 	# Compute the squared difference between observed and predicted
# 	def propDiff(f):
# 		diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))
#
# 		return f.set('diff', diff.pow(2))
#
# 	# calculate RMSE from squared difference
# 	rmse = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean')).sqrt()
#
# 	return rmse
#
# # MAE function
# def MAE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
# 	# Compute the absolute difference between observed and predicted
# 	def propDiff(f):
# 		diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))
#
# 		return f.set('diff', diff.abs())
#
# 	# calculate RMSE from squared difference
# 	mae = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean'))
#
# 	return mae
#
# # Function to add folds stratified per biome
# def assignFolds(biome):
# 	fc_filtered = fc_agg.filter(ee.Filter.eq(stratificationVariableString, biome))
#
# 	cvFoldsToAssign = ee.List.sequence(0, fc_filtered.size()).map(lambda i: ee.Number(i).mod(k).add(1))
#
# 	fc_sorted = fc_filtered.randomColumn(seed = biome).sort('random')
#
# 	fc_wCVfolds = ee.FeatureCollection(cvFoldsToAssign.zip(fc_sorted.toList(fc_filtered.size())).map(lambda f: ee.Feature(ee.List(f).get(1)).set(cvFoldString, ee.List(f).get(0))))
#
# 	return fc_wCVfolds
#
#
# # Define a function to take a feature with a classifier of interest
# def computeCVAccuracyAndRMSE(featureWithClassifier):
# 	# Pull the classifier from the feature
# 	cOI = ee.Classifier(featureWithClassifier.get('c'))
#
# 	# Create a function to map through the fold assignments and compute the overall accuracy
# 	# for all validation folds
# 	def computeAccuracyForFold(foldFeature):
# 		# Organize the training and validation data
# 		foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
# 		trainingData = fcOI.filterMetadata(cvFoldString,'not_equals',foldNumber)
# 		validationData = fcOI.filterMetadata(cvFoldString,'equals',foldNumber)
#
# 		# Train the classifier and classify the validation dataset
# 		trainedClassifier = cOI.train(trainingData,classProperty,covariateList)
# 		outputtedPropName = classProperty+'_Predicted'
# 		classifiedValidationData = validationData.classify(trainedClassifier,outputtedPropName)
#
# 		# Compute accuracy metrics
# 		r2ToSet = coefficientOfDetermination(classifiedValidationData,classProperty,outputtedPropName)
# 		rmseToSet = RMSE(classifiedValidationData,classProperty,outputtedPropName)
# 		maeToSet = MAE(classifiedValidationData,classProperty,outputtedPropName)
# 		return foldFeature.set('R2',r2ToSet).set('RMSE', rmseToSet).set('MAE', maeToSet)
#
# 	# Compute the mean and std dev of the accuracy values of the classifier across all folds
# 	accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)
# 	meanAccuracy = accuracyFC.aggregate_mean('R2')
# 	sdAccuracy = accuracyFC.aggregate_total_sd('R2')
#
# 	# Calculate mean and std dev of RMSE
# 	RMSEvals = accuracyFC.aggregate_array('RMSE')
# 	RMSEvalsSquared = RMSEvals.map(lambda f: ee.Number(f).multiply(f))
# 	sumOfRMSEvalsSquared = RMSEvalsSquared.reduce(ee.Reducer.sum())
# 	meanRMSE = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared).divide(k))
#
# 	sdRMSE = accuracyFC.aggregate_total_sd('RMSE')
#
# 	# Calculate mean and std dev of MAE
# 	meanMAE = accuracyFC.aggregate_mean('MAE')
# 	sdMAE= accuracyFC.aggregate_total_sd('MAE')
#
# 	# Compute the feature to return
# 	featureToReturn = featureWithClassifier.select(['cName']).set('Mean_R2',meanAccuracy,'StDev_R2',sdAccuracy, 'Mean_RMSE',meanRMSE,'StDev_RMSE',sdRMSE, 'Mean_MAE',meanMAE,'StDev_MAE',sdMAE)
# 	return featureToReturn
#
#






##################################################################################################################################################################
# Univariate int-ext analysis
##################################################################################################################################################################
# Univariate interpolation/extrapolation helper function
def assessUnivarExtrapolation(fcOI):
	# Create a feature collection with only the values from the image bands
	fcForMinMax = fcOI.select(covariateList)

	# Make a FC with the band names
	fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(ee.Geometry.Point([0,0])).set('BandName',bandName)))

	def calcMinMax(f):
  		bandBeingComputed = f.get('BandName')
  		maxValueToSet = fcForMinMax.reduceColumns(ee.Reducer.minMax(),[bandBeingComputed])
  		return f.set('MinValue',maxValueToSet.get('min')).set('MaxValue',maxValueToSet.get('max'))

	# Map function
	fcWithMinMaxValues = ee.FeatureCollection(fcWithBandNames).map(calcMinMax)

	# Make two images from these values (a min and a max image)
	maxValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MaxValue'))
	maxDict = ee.Dictionary.fromLists(covariateList,maxValuesWNulls)
	minValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MinValue'))
	minDict = ee.Dictionary.fromLists(covariateList,minValuesWNulls)
	minImage = minDict.toImage()
	maxImage = maxDict.toImage()

	totalBandsBinary = compositeToClassify.gte(minImage.select(covariateList)).lt(maxImage.select(covariateList))
	univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(compositeToClassify.bandNames().length()).rename('univariate_pct_int_ext')
	return univariate_int_ext_image

# Create ee.ImageCollection with univariate interpolation/extrapolation images
def univarExtrapolationIC(start, listOfAssetIds):
	# Define the start and end of the bootstrapped images
	start = start
	end = start + 99

	# Get the assetIds of the bootstrapped data
	listOfBootDatIds = listOfAssetIds[start:end]

	# Loop over the bootstrapped data and return the univariate interpolation/extrapolation image
	listOfIntExtImages = []
	for bootDatId in listOfBootDatIds:
		listOfIntExtImages.append(ee.Image(assessUnivarExtrapolation(ee.FeatureCollection(bootDatId))))

	# Get the univariate interpolation/extrapolation percentage
	univariateMeanStdDevImage = ee.ImageCollection(listOfIntExtImages).reduce(reducer = ee.Reducer.mean().combine(reducer2 = ee.Reducer.stdDev(),sharedInputs=True)).rename(['univariate_pct_int_ext_mean','univariate_pct_int_ext_stddev'])
	univariateMeanStdDevImageExport = ee.batch.Export.image.toAsset(
	    image = univariateMeanStdDevImage,
	    description = 'UnivariateIntExt_' + str(start) + '_' + str(end+1),
	    assetId = projectId + 'IntExt_univariate/UnivariateIntExt_' + str(start) + '_' + str(end+1),
	    crs = 'EPSG:4326',
	    crsTransform = compositeToClassify.projection().getInfo().get('transform'),
	    region = unboundedGeo,
	    maxPixels = int(1e13))
	univariateMeanStdDevImageExport.start()


##################################################################################################################################################################
# Multivariate (PCA) int-ext analysis
##################################################################################################################################################################
# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
	result = []
	# Fetch data as a list
	values = fc.toList(100000).getInfo()
	# Fetch column names
	BANDS = fc.first().propertyNames().getInfo()
	# Remove system:index if present
	if 'system:index' in BANDS: BANDS.remove('system:index')

	# Convert to data frame
	for item in values:
		values = item['properties']
		row = [str(values[key]) for key in BANDS]
		row = ",".join(row)
		result.append(row)

	df = pd.DataFrame([item.split(",") for item in result], columns = BANDS)
	df.replace('None', np.nan, inplace = True)

	return df

# PCA interpolation/extrapolation helper function
def assessExtrapolation(fcOfInterest, propOfVariance):
	# Transform ee.FeatureCollection into pd dataframe
	fcOfInterest = GEE_FC_to_pd(fcOfInterest)[covariateList.getInfo()]

	# Compute the mean and standard deviation of each band, then standardize the point data
	meanVector = fcOfInterest.mean()
	stdVector = fcOfInterest.std()
	standardizedData = (fcOfInterest-meanVector)/stdVector

	# Then standardize the composite from which the points were sampled
	meanList = meanVector.tolist()
	stdList = stdVector.tolist()
	bandNames = list(meanVector.index)
	meanImage = ee.Image(meanList).rename(bandNames)
	stdImage = ee.Image(stdList).rename(bandNames)
	standardizedImage = compositeToClassify.subtract(meanImage).divide(stdImage)

	# Run a PCA on the point samples
	pcaOutput = PCA()
	pcaOutput.fit(standardizedData)

	# Save the cumulative variance represented by each PC
	cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4)*100)

	# Make a list of PC names for future organizational purposes
	pcNames = ['PC'+str(x) for x in range(1,fcOfInterest.shape[1]+1)]

	# Get the PC loadings as a data frame
	loadingsDF = pd.DataFrame(pcaOutput.components_,columns=[str(x)+'_Loads' for x in bandNames],index=pcNames)

	# Get the original data transformed into PC space
	transformedData = pd.DataFrame(pcaOutput.fit_transform(standardizedData,standardizedData),columns=pcNames)

	# Make principal components images, multiplying the standardized image by each of the eigenvectors
	# Collect each one of the images in a single image collection

	# First step: make an image collection wherein each image is a PC loadings image
	listOfLoadings = ee.List(loadingsDF.values.tolist())
	eePCNames = ee.List(pcNames)
	zippedList = eePCNames.zip(listOfLoadings)
	def makeLoadingsImage(zippedValue):
		return ee.Image.constant(ee.List(zippedValue).get(1)).rename(bandNames).set('PC',ee.List(zippedValue).get(0))
	loadingsImageCollection = ee.ImageCollection(zippedList.map(makeLoadingsImage))

	# Second step: multiply each of the loadings image by the standardized image and reduce it using a "sum"
	# to finalize the matrix multiplication
	def finalizePCImages(loadingsImage):
		PCName = ee.String(ee.Image(loadingsImage).get('PC'))
		return ee.Image(loadingsImage).multiply(standardizedImage).reduce('sum').rename([PCName]).set('PC',PCName)
	principalComponentsImages = loadingsImageCollection.map(finalizePCImages)

	# Choose how many principal components are of interest in this analysis based on amount of
	# variance explained
	numberOfComponents = sum(i < propOfVariance for i in cumulativeVariance)+1
	print('Number of Principal Components being used:',numberOfComponents)

	# Compute the combinations of the principal components being used to compute the 2-D convex hulls
	tupleCombinations = list(combinations(list(pcNames[0:numberOfComponents]),2))
	print('Number of Combinations being used:',len(tupleCombinations))

	# Generate convex hulls for an example of the principal components of interest
	cHullCoordsList = list()
	for c in tupleCombinations:
		firstPC = c[0]
		secondPC = c[1]
		outputCHull = ConvexHull(transformedData[[firstPC,secondPC]])
		listOfCoordinates = transformedData.loc[outputCHull.vertices][[firstPC,secondPC]].values.tolist()
		flattenedList = [val for sublist in listOfCoordinates for val in sublist]
		cHullCoordsList.append(flattenedList)

	# Reformat the image collection to an image with band names that can be selected programmatically
	pcImage = principalComponentsImages.toBands().rename(pcNames)

	# Generate an image collection with each PC selected with it's matching PC
	listOfPCs = ee.List(tupleCombinations)
	listOfCHullCoords = ee.List(cHullCoordsList)
	zippedListPCsAndCHulls = listOfPCs.zip(listOfCHullCoords)

	def makeToClassifyImages(zippedListPCsAndCHulls):
		imageToClassify = pcImage.select(ee.List(zippedListPCsAndCHulls).get(0)).set('CHullCoords',ee.List(zippedListPCsAndCHulls).get(1))
		classifiedImage = imageToClassify.rename('u','v').classify(ee.Classifier.spectralRegion([imageToClassify.get('CHullCoords')]))
		return classifiedImage

	classifedImages = ee.ImageCollection(zippedListPCsAndCHulls.map(makeToClassifyImages))
	finalImageToExport = classifedImages.sum().divide(ee.Image.constant(len(tupleCombinations)))

	return finalImageToExport

# Create ee.ImageCollection with PCA interpolation/extrapolation images
def pcaExtrapolationIC(start, listOfAssetIds, propOfVariance):
	# Define the start and end of the bootstrapped images
	start = start
	end = start + 99

	# Get the assetIds of the bootstrapped data
	listOfBootDatIds = listOfAssetIds[start:end]

	# Loop over the bootstrapped data and return the univariate interpolation/extrapolation image
	listOfIntExtImages = []
	for bootDatId in listOfBootDatIds:
		listOfIntExtImages.append(ee.Image(assessExtrapolation(ee.FeatureCollection(bootDatId),propOfVariance)))

	# Get the univariate interpolation/extrapolation percentage
	pcaMeanStdDevImage = ee.ImageCollection(listOfIntExtImages).reduce(reducer = ee.Reducer.mean().combine(reducer2 = ee.Reducer.stdDev(),sharedInputs=True)).rename(['pca_pct_int_ext_mean','pca_pct_int_ext_stddev'])
	pcaMeanStdDevImageExport = ee.batch.Export.image.toAsset(
	    image = pcaMeanStdDevImage,
	    description = 'PCAIntExt_' + str(start) + '_' + str(end+1),
	    assetId = projectId + 'IntExt_pca/PCAIntExt_' + str(start) + '_' + str(end+1),
	    crs = 'EPSG:4326',
	    crsTransform = compositeToClassify.projection().getInfo().get('transform'),
	    region = unboundedGeo,
	    maxPixels = int(1e13))
	pcaMeanStdDevImageExport.start()

##################################################################################################################################################################
### Run the extrapolation functions

# Get the assetIds of the bootstrapped data
listOfAssetIds = subprocess.run(bashCommandList_List + [folderBootstrappedData],stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()

# Get the univariate interpolation/extrapolation percentage
list(map(partial(univarExtrapolationIC, listOfAssetIds = listOfAssetIds),range(0,500,100)))

# Get the PCA interpolation/extrapolation percentage
list(map(partial(pcaExtrapolationIC, listOfAssetIds = listOfAssetIds, propOfVariance = propOfVariance),range(0,500,100)))
