
// Load the results from the BLOO CV
var stringToUse = 'Rarity_bloo_cv_';
var listOfIndices = ee.List.sequence(0,9);

// Create a client-side list that holds one element less than the # of FeatureCollections to export 
var listForExport = [0,1,2,3,4,5,6,7,8,9];

// Map over the list to export
listForExport.map(function(n){
  
  // Get the description
  var description = ee.String(stringToUse).cat(ee.Number(n).add(1).format('%02d')).getInfo();
  
  // Get the FeatureCollection
  var loadedFC = ee.FeatureCollection('users/laubert/DominanceAndRarity/Bloo_CV_Results/' + description);
  
  // Export the image
  Export.table.toDrive({
    collection: loadedFC,
    description: description,
    folder:'GEE_OUTPUT'
  });
});