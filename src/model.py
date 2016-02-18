import os
import numpy as np
from random import randint
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from src.utils import comma_seperated_to_vector, get_feature_file, create_submission_file

"""

We provide a model for developing a driver's fingerprint based on GPS data of their drives.

In this case the model we use:

clfs = [RandomForestClassifier(n_estimators=3000,min_samples_split=1, n_jobs=-1)]

"""

"""configuration constants"""
CSV_NAME = "/3.11.15-featureGeneration.csv"
numOfFeatures = 123
numOfDrivesInFolder = 200
numOfRandomExamples = 1000


def createFeatureVector(line):
    """Creates a vector of features for each trip."""
    features = []
    features.append(comma_seperated_to_vector(line))
    features = features[0]
    featuresList = []
    for a in features:
        featuresList.append(float(a))
    return featuresList

def getTripMatchArray(tripMatchLines, beg, end):
    tripMatches = []
    for a in tripMatchLines[beg-1:end-1]:
        a = a.replace("\n", "")
        a = a.split(",")
        tripMatches.append(int(a[2]))
    return tripMatches


def getTripLabels(testData):
    tripLabels = []
    for a in testData:
        tripLabels.append(a[0])
    return np.array(tripLabels)


def getTrainingLabels(numOfRandomExamples, numOfDrivesInFolder):
    bad = np.zeros((numOfRandomExamples,))
    good = np.ones((numOfDrivesInFolder, ))
    return np.concatenate((good, bad))


def readFile(featuresFile):
    with open(featuresFile, "r") as featureFile:
        return featureFile.readlines()


def modelAnalysis(feature_path, submission_path, numOfFeatures, end):
        #open submission file.
        with open(create_submission_file(submission_path), 'w') as writeFile:
            write_header(writeFile)
            lines = readFile(feature_path)
            negativeTrainingData = vectorOfFalseTrainingData(lines, numOfFeatures)
            #create feature matrix
            for driver in range(1,end):
                begin =  (driver -1)*numOfDrivesInFolder+1
                end = (driver *numOfDrivesInFolder)+1
                driverNumber = int(lines[((driver-1)*numOfDrivesInFolder)+1].split(",")[0][1:].strip("'"))
                count = 1
                featureMatrix = np.zeros((numOfFeatures, ))
                for b in lines[((driver-1)*numOfDrivesInFolder)+1:(driver*numOfDrivesInFolder) + 1]:
                    a = np.array(createFeatureVector(b))
                    featureMatrix = np.vstack([featureMatrix, a])
                featureMatrix = np.delete(featureMatrix, (0), axis=0)
                testData = featureMatrix
                testData[np.isnan(testData)] = 0

                #Create array of trip labels (used for output)
                tripLabels = getTripLabels(testData)
                #Create array of training labels (Currently: 200 1s and numOfRandomExamples 0s)
                trainingLabels = getTrainingLabels(numOfRandomExamples, numOfDrivesInFolder)

                #Make DType
                featureMatrix[np.isnan(featureMatrix)]=0
                trainingLabels[np.isnan(trainingLabels)]=0
                negativeTrainingData[np.isnan(featureMatrix)] =0
                featureMatrix = np.vstack([featureMatrix, negativeTrainingData])
                tripLabels = tripLabels.astype(np.int, copy = False)
                trainingLabels = trainingLabels.astype(np.float32, copy=False)
                featureMatrix = featureMatrix.astype(np.float32, copy=False)
                featureMatrix[np.isinf(featureMatrix)]=0
                testData[np.isinf(testData)] = 0
                featureMatrix[np.isnan(featureMatrix)]=0

                clfs = [RandomForestClassifier(n_estimators=3000,min_samples_split=1, n_jobs=-1)]

                predicted_list = []

                for ExecutionIndex, clf in enumerate(clfs):
                    clf.fit(featureMatrix, trainingLabels)
                    predicted_probs = clf.predict_proba(testData)[:,1]
                    predicted_list.append(predicted_probs)
                predicted_list = np.average(predicted_list, axis=0)
                count = 0
                print predicted_list
                for a in predicted_list:
                    returnString = "%d_%d,%.3f" % (driverNumber, tripLabels[count], a)
                    writeFile.write("%s\n" % returnString)
                    print returnString
                    count = count+1


def vectorOfFalseTrainingData(lines, numOfFeatures):
    negativeTrainingData = np.zeros((numOfFeatures,))
    for randomNumbers in range(0, numOfRandomExamples):
        randomNumber = randint(1, len(lines) - 1)
        a = np.array(createFeatureVector(lines[randomNumber]))
        negativeTrainingData = np.vstack([negativeTrainingData, a])
    np.delete(negativeTrainingData, (0), axis=0)
    return negativeTrainingData


def write_header(writefile):
    writefile.write("driver_trip,prob\n")


if __name__ == '__main__':
    submissionPath, featuresFile = get_feature_file(CSV_NAME)
    modelAnalysis(submissionPath, featuresFile ,numOfFeatures, end = 2737)
