
## AssessModelOutputs.r
# This script contains code to assess the cross validated results from GEE
# by calculating the R^2 values across the cross validation folds

# Variable of interest
varOfInterest <- 'Rarity'

# ————————————————————————————————————————————————————————————————————————————————
# Load necessary packages (install with "install.packages(c('package_name_goes_here'))")
# Or use "Tools / Install Packages..." within RStudio
library(dplyr) # The "tidyverse" is a used for organizing and wrangling data
library(matrixStats) # "matrixStats" is used to calculate row standard deviations

# !! Set the working directory to a specified folder containing the data of interest
# (i.e., where the data is located and where you want to output any files)
setwd('/Users/Thomas/Projects/DominanceRarity/output')

# Create a "population standard deviation" function (the sd() function in R computes
# "sample standard deviation")
popSD <- function(x){sd(x)*sqrt((length(x)-1)/length(x))}

# ————————————————————————————————————————————————————————————————————————————————
# Input the file path and load the data from the CSV as a data frame
resultsPath <- paste0('Predicted_vs_Observed_',varOfInterest,'.csv')
resultsDataFrame <- read.csv(file=resultsPath, header=TRUE)

# Subset only the columns of interest
originalDependentVariable <- varOfInterest
modelledValue <- paste0(varOfInterest,'_mean')
# !! These strings should be adjusted according to the actual column names
resultsDataFrame <- resultsDataFrame[,c(originalDependentVariable,modelledValue)]
str(resultsDataFrame)

# ————————————————————————————————————————————————————————————————————————————————
# Split the data by fold number
# listOfFolds <- split(resultsDataFrame, resultsDataFrame$Fold_Number)

# The dataframe is split into a list of dataframes, with the same number
# of elements as cross validation folds
# print(listOfFolds)

# ————————————————————————————————————————————————————————————————————————————————
# Compute the R^2 for each fold by using the sapply() function (which applies a function
# of interest to every element of a list)
# !! The "Observed_Column" and "Predicted_Column" should be changed to the same values
# !! as inputted in the column subsetting above (except without quotation marks)
computeR2 <- function(dataFrame) (1 - (sum((dataFrame[,c(originalDependentVariable)]-dataFrame[,c(modelledValue)])^2)/sum((dataFrame[,c(originalDependentVariable)]-mean(dataFrame[,c(originalDependentVariable)]))^2)))
cat(paste('CV R^2 Value\n', computeR2(resultsDataFrame)))
# cvResultsPerFold

# ————————————————————————————————————————————————————————————————————————————————
# ANOVA test
dat <- bind_rows(r2values, .id = "column_label")
dat <- cbind(dat, buffer = bufferSizes)
dat[,'buffer_size'] <- as.factor(dat[,'buffer_size'])

res.aov <- aov(R2_val ~ buffer_size, data = dat)
summary(res.aov)
test <- TukeyHSD(res.aov)
test

# ————————————————————————————————————————————————————————————————————————————————
# Get the block leave-one-out CV values
setwd('BLOOCV')
resultsPaths <- list.files(pattern = originalDependentVariable)
listOfData <- lapply(resultsPaths, read.csv)
r2values <- lapply(listOfData, `[`, 2)
bufferSizes <- lapply(listOfData, `[`, 3)[1]

# Compute the mean of folds 
data_mean <- data.frame(x = bufferSizes, R2_mean = rowMeans(do.call(cbind, r2values)), R2_sd = apply(do.call(cbind, r2values),1, sd))
data_mean
