import pandas as pd
import numpy as np
import json

##############################################################################################################################
##############################################################################################################################
### DATA AGGREGATION

sampled_data = pd.read_csv('../data/areaOfBiomesPerCountry.csv', dtype=object, encoding='latin', index_col='system:index')
sampled_data = sampled_data.drop(columns='.geo')



# Get the unique admin unit ids
adminUnits = sorted(list(set(sampled_data['GID_0'])))

# Define the parameters that give the mean or the mode over an administrative unit
parameters = ['biome01','biome02','biome03','biome04','biome05','biome06','biome07','biome08','biome09','biome10','biome11','biome12','biome13','biome14','totalArea']
staticParameters = list(['NAME_0'])

# Define the export data frame
exportData = pd.DataFrame(columns=sampled_data.columns, index=range(len(adminUnits)))
exportData['GID_0'] = adminUnits

# Loop over the admin units
for unit in adminUnits:
    # Select each unit
    dat = sampled_data.loc[sampled_data['GID_0'] == unit]
    # Loop over the sum parameters
    for sP in parameters:
        # Get the parameter
        sP_dat = dat[[sP]].astype(np.float64)
        # Append the sum of the parameter
        exportData.at[exportData[exportData['GID_0']==unit].index.values[0],sP] = sP_dat.sum()[sP]

    # Add the other parameters
    for sP in staticParameters:
        # Add the static parameters to the export dataframe
        exportData.at[exportData[exportData['GID_0']==unit].index.values[0], sP] = dat[staticParameters].iloc[0][sP]


# Select the columns of interest
exportData = exportData[['GID_0','NAME_0'] + parameters]

# Rename the columns
exportData = exportData.add_suffix('_areaInKm2')
exportData.rename(columns={'GID_0_areaInKm2': 'GID_0', 'NAME_0_areaInKm2':'NAME'},inplace=True)

# Compute percentages
exportData['biome01_pct'] = (exportData['biome01_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome02_pct'] = (exportData['biome02_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome03_pct'] = (exportData['biome03_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome04_pct'] = (exportData['biome04_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome05_pct'] = (exportData['biome05_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome06_pct'] = (exportData['biome06_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome07_pct'] = (exportData['biome07_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome08_pct'] = (exportData['biome08_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome09_pct'] = (exportData['biome09_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome10_pct'] = (exportData['biome10_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome11_pct'] = (exportData['biome11_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome12_pct'] = (exportData['biome12_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome13_pct'] = (exportData['biome13_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100
exportData['biome14_pct'] = (exportData['biome14_areaInKm2'] / exportData['totalArea_areaInKm2']) * 100

# Export the data
exportData.to_csv('../data/GADM36_biomeAreas.csv')
