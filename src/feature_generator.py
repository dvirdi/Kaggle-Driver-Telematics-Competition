from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

author = 'daljeetv'
import os
from math import hypot
import numpy as numpy
import pandas as pd
from scipy import signal
global strt
strt = 0

"""

This file generates a list of features for a given list of gps coordinates.

The features include velocity,acceleration,
jerk, characteristics of the road, centripetal acceleration,
and numerous combinations and percentiles thereof.

"""

def getFeatures(x, y, t1, a, b):

        """this is the primary method that generates a feature_vector"""

        #Pre-process/Clean Data
        xn, yn, velX, velY, numberOfRepetitions = preProcessing(t1)
        #Length Of Trip
        triplength = abs(distance(xn[0], yn[0], xn[-1], yn[-1]))

        #Velocity
        velX = numpy.array(numpy.diff(xn,n=1))
        velY = numpy.array(numpy.diff(yn, n=1))
        #Centripetal Acceleration
        centAcc = []
        for count in range(0,len(velX)):
            magOfB = pow(velX[count],2) + pow(velY[count],2)
            centAcc.append((velX[count-1] * velY[count-1] + velX[count] * velY[count]) % magOfB)
        rangeOfCentAcc = abs(numpy.percentile(centAcc,98) - numpy.percentile(centAcc,2))
        maxCentAcc = max(centAcc)
        #Distances
        v, distancecovered, numberOfHyperSpaceJumps = velocities_and_distance_covered(xn, yn)
        farthestxPos = numpy.max(xn)
        farthestxNeg = numpy.min(xn)

        yn80 = numpy.percentile(yn, 80)
        yn90 = numpy.percentile(yn, 90)

        ynReal100, ynReal25, ynReal50, ynReal75, ynReal80 = how_far_by_percentage_of_trip(xn)

        ynReal100, ynReal25, ynReal50, ynReal75, ynReal80 = how_far_by_percentage_of_trip(yn)

        distanceCoveredFirst25, distanceCoveredFirst50_25, distanceCoveredFirst50_75, distanceCoveredFirst75_100 = distance_covered(
            xnReal100, xnReal25, xnReal50, xnReal75, ynReal100, ynReal25, ynReal50, ynReal75)


        farthestx = max(abs(farthestxPos), abs(farthestxPos))

        numberOfPointsAfterFarthestX = 0
        flag = False
        for pointsX in xn:
            if(pointsX == farthestx):
                flag = True
            if flag:
                numberOfPointsAfterFarthestX = numberOfPointsAfterFarthestX +1

        farthesty, farthestyNeg, farthestyPos = geography_of_plane(yn)

        numberOfPointsAfterFarthestY = 0
        flag = False
        for pointsY in yn:
            if(pointsY == farthesty):
                flag = True
            if flag:
                numberOfPointsAfterFarthestY = numberOfPointsAfterFarthestY +1
        #Velocity
        maxspeed = numpy.max(v)
        totalSpeed = numpy.sum(v)
        rangeOfVel = abs(numpy.max(v) - numpy.min(v))

        gpsLag, speed30, speed70, speed90, speed99, speedingAbove10Below15, speedingAbove15Below20, speedingAbove20, speedingBelow_3 = speed_features(
            v)

        averageChangeInVelocity = 0
        count = 0
        maxVelly = -1000
        minVelly = 1000
        ranges = []
        for vel in v:
            if(count < 20):
                count += 1
                if(vel < minVelly):
                    minVelly = vel
                if(vel > maxVelly):
                    maxVelly = vel
            else:
                if(abs(minVelly - maxVelly) != 2000):
                    ranges.append(abs(minVelly - maxVelly))
                minVelly = 1000
                maxVelly = -1000
                count = 0
        #ranges = numpy.array(ranges)
        ranges = numpy.sort(ranges)
        if(len(ranges) > 0):
            highestRange = ranges[-1]
            lowestRange = ranges[0]
            rangesAverage = numpy.average(ranges)
        else:
            highestRange = 0
            lowestRange = 0
            rangesAverage = 0

        #Acceleration
        accelerationvector, accArray, decArray, numOfChanges= acceleration(v)

        accelAbove2, accelAbove_2Below0, accelAbove_5, accelBelow2Above0, accelBelow_2, constant, constantMax = accelerationOverCertainValues(
            accelerationvector)

        acc90 = numpy.percentile(accArray, 90)
        rangeOfAcc = abs(numpy.max(accelerationvector) - numpy.min(accelerationvector))
        maxAcc = numpy.max(accArray)
        totalAcc = numpy.sum(accArray)
        maxDec = numpy.min(decArray)
        totalDec = numpy.sum(decArray)
        dec01 = numpy.percentile(decArray, 0.1)

        vAccTupleSortedByAcceleration = zip(v, accelerationvector)
        vAccTupleSortedByVelocity = zip(v, accelerationvector)
        vAccTupleSortedByAcceleration.sort(key=lambda vAccTuple: vAccTuple[1])

        averageVelFor20Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[0:20]])
        averageVelFor40Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[20:40]])
        averageVelFor60Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[40:60]])
        averageVelFor80Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[60:80]])
        averageVelFor100Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[80:100]])
        averageVelFor120Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[100:120]])
        averageVelFor140Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[120:140]])

        averageVelFor10Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[0:10]])
        averageVelFor220Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[10:20]])
        averageVelFor30Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[20:30]])
        averageVelFor440Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[30:40]])
        averageVelFor50Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[40:50]])
        averageVelFor660Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[50:60]])
        averageVelFor70Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[60:70]])
        averageVelFor880Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[70:80]])
        averageVelFor90Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[80:90]])
        averageVelFor1100Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[90:100]])
        averageVelFor1110Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[100:110]])
        averageVelFor1120Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[110:120]])
        averageVelFor130Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[120:130]])
        averageVelFor1140Acc =  numpy.average([x for x,_ in vAccTupleSortedByAcceleration[130:140]])

        #Jerk
        jerk, negjerk, posjerk = jerk(accelerationvector)
        averageOfNegativeJerks, averageOfPositiveJerks, jerk90, negjerk10, negjerk30, negjerk70, negjerk90, posjerk10, posjerk30, rangeOfJerk = statistics_of_jerk(
            jerk, negjerk, posjerk)
        #jerk averages
        vVelTupleSortedByJerk = acceleration_sorted_by_jerk(accelerationvector, jerk, v)
        averageVelFor100Jerk, averageVelFor120Jerk, averageVelFor140Jerk, averageVelFor200Jerk, averageVelFor20Jerk, averageVelFor40Jerk, averageVelFor60Jerk, averageVelFor80Jerk = percentiles_of_jerk(
            vVelTupleSortedByJerk)

        #Tangent
        maxx, maxy, tangent, tangent0_25, tangent0_50, tangent20_50, tangent50100, tangent50_80, tangent70_100 = tangents(
            farthestxNeg, farthestxPos, farthestyNeg, farthestyPos, xnReal100, xnReal25, xnReal50, xnReal75, xnReal80,
            ynReal100, ynReal25, ynReal50, ynReal75, ynReal80)

        #Theta Basics
        theta = curvature_spline(xn, yn)

        velatAngleAbove1Below5, velatAngleAbove5, velatAngleBelow1Above_1, velatAngleBelow_10, velatAngleBelow_1Above_5, velatAngleBelow_5Above_10 = curvature_at_velocities(
            theta, v)

        avgvelatAngleAbove1Below5, avgvelatAngleAbove5, avgvelatAngleBelow1Above_1, avgvelatAngleBelow_10, avgvelatAngleBelow_1Above_5, avgvelatAngleBelow_5Above_10 = avg_velocity_at_angles(
            velatAngleAbove1Below5, velatAngleAbove5, velatAngleBelow1Above_1, velatAngleBelow_10,
            velatAngleBelow_1Above_5, velatAngleBelow_5Above_10)

        accatAngleAbove1Below5, accatAngleAbove5, accatAngleBelow1Above_1, accatAngleBelow_10, accatAngleBelow_1Above_5, accatAngleBelow_5Above_10 = acceleration_at_angles(
            accelerationvector, theta)

        avgvelatAngleAbove1Below5, avgvelatAngleAbove5, avgvelatAngleBelow1Above_1, avgvelatAngleBelow_10, avgvelatAngleBelow_1Above_5, avgvelatAngleBelow_5Above_10 = avg_velocity_at_angles(
            accatAngleAbove1Below5, accatAngleAbove5, accatAngleBelow1Above_1, accatAngleBelow_10,
            accatAngleBelow_1Above_5, accatAngleBelow_5Above_10)

        rangeOfTheta = abs(numpy.percentile(theta,99) - numpy.percentile(theta,1))

        #Theta/Angles
        tangentActual, theta2, theta25, theta50, theta75, theta90, theta98 = theta_percentiles(maxx, maxy, theta)

        anglAbove0, anglAbove0Below0_5, anglAbove0_5Below10, anglAbove10, anglAboveAbs0_5, anglAboveNeg0_5Below0, anglAboveNeg1Below0_5, anglBelowAbs0_5, anglBelowNeg1, angleBelow0 = angle_features(
            theta)

        #longest stretch of straight road
        lengthOfEveryStraightSegments, lengthOfEveryTurnSegment = length_of_straight_roads(theta)

        lengthOfEveryStraightSegments = numpy.sort(numpy.unique(numpy.array(lengthOfEveryStraightSegments)))
        lengthOfEveryTurnSegment = numpy.sort(numpy.unique(numpy.array(lengthOfEveryTurnSegment)))

        curveLine1, straightLine1, straightLine2 = get_major_roads(lengthOfEveryStraightSegments,
                                                                   lengthOfEveryTurnSegment)

        """ build feature vector"""
        features = []

        features.append(a)
        features.append(b)

        #Distance
        features.append(triplength)
        features.append(farthestx)
        features.append(farthesty)
        features.append(numberOfPointsAfterFarthestY)
        features.append(numberOfPointsAfterFarthestX)
        features.append(numberOfRepetitions)
        features.append(yn80)
        features.append(yn90)
        features.append(distanceCoveredFirst25)
        features.append(distanceCoveredFirst50_25)
        features.append(distanceCoveredFirst50_75)
        features.append(distanceCoveredFirst75_100)

        #Speed
        features.append(totalSpeed)
        features.append(maxspeed)
        features.append(speedingBelow_3)
        features.append(speedingAbove20)
        features.append(speedingAbove15Below20)
        features.append(speedingAbove10Below15)

        features.append(averageVelFor20Acc)
        features.append(averageVelFor40Acc)
        features.append(averageVelFor60Acc)
        features.append(averageVelFor80Acc)
        features.append(averageVelFor100Acc)
        features.append(averageVelFor120Acc)
        features.append(averageVelFor140Acc)
        features.append(averageVelFor20Jerk)
        features.append(averageVelFor40Jerk)
        features.append(averageVelFor60Jerk)
        features.append(averageVelFor80Jerk)
        features.append(averageVelFor100Jerk)
        features.append(averageVelFor120Jerk)
        features.append(averageVelFor140Jerk)
        features.append(averageVelFor200Jerk)

        #Acceleration
        features.append(numOfChanges)
        features.append(maxAcc)
        features.append(maxDec)
        features.append(totalAcc)
        features.append(totalDec)
        features.append(acc90)
        features.append(dec01)
        features.append(averageOfPositiveJerks)
        features.append(averageOfNegativeJerks)
        features.append(accelAbove2)
        features.append(accelBelow2Above0)
        features.append(accelAbove_2Below0)
        features.append(accelBelow_2)
        features.append(accelAbove_5)
        features.append(constantMax)
        features.append(constant)

        #Tangents
        features.append(tangent)
        features.append(tangentActual)
        features.append(tangent50100)
        features.append(tangent0_50)
        features.append(tangent0_25)
        features.append(tangent20_50)
        features.append(tangent50_80)
        features.append(tangent70_100)

        #Theta
        features.append(theta98)
        features.append(theta90)
        features.append(theta75)
        features.append(theta50)
        features.append(theta25)
        features.append(theta2)

        #Speed
        features.append(speed99)
        features.append(speed90)
        features.append(speed70)
        features.append(speed30)
        features.append(highestRange)
        features.append(lowestRange)
        features.append(rangesAverage)


        features.append(jerk90)
        features.append(posjerk30)
        features.append(posjerk10)
        features.append(negjerk90)
        features.append(negjerk70)
        features.append(negjerk30)
        features.append(negjerk10)

        features.append(averageVelFor10Acc)
        features.append(averageVelFor220Acc)
        features.append(averageVelFor30Acc)
        features.append(averageVelFor440Acc)
        features.append(averageVelFor50Acc)
        features.append(averageVelFor660Acc)
        features.append(averageVelFor70Acc)
        features.append(averageVelFor880Acc)
        features.append(averageVelFor90Acc)
        features.append(averageVelFor1100Acc)
        features.append(averageVelFor1110Acc)
        features.append(averageVelFor1120Acc)
        features.append(averageVelFor130Acc)
        features.append(averageVelFor1140Acc)

        features.append(straightLine1)
        features.append(straightLine2)
        features.append(curveLine1)

        #Misc
        features.append(gpsLag)
        features.append(maxCentAcc)

        features.append(avgvelatAngleBelow_10)
        features.append(avgvelatAngleBelow_5Above_10)
        features.append(avgvelatAngleBelow_1Above_5)
        features.append(avgvelatAngleBelow1Above_1)
        features.append(avgvelatAngleAbove1Below5)
        features.append(avgvelatAngleAbove5)

        features.append(avgaccatAngleBelow_10)
        features.append(avgaccatAngleBelow_5Above_10)
        features.append(avgaccatAngleBelow_1Above_5)
        features.append(avgaccatAngleBelow1Above_1)
        features.append(avgaccatAngleAbove1Below5)
        features.append(avgaccatAngleAbove5)

        #Ranges
        features.append(rangeOfAcc)
        features.append(rangeOfJerk)
        features.append(rangeOfTheta)
        features.append(rangeOfVel)

        features.append(angleBelow0)
        features.append(anglAbove0)
        features.append(anglAbove10)
        features.append(anglAbove0_5Below10)
        features.append(anglAbove0Below0_5)
        features.append(anglAboveNeg0_5Below0)
        features.append(anglAboveNeg1Below0_5)
        features.append(anglBelowNeg1)
        features.append(anglAboveAbs0_5)
        features.append(anglBelowAbs0_5)
        return str(features)



def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))

def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = []
    numberOfHyperSpaceJumps = 0
    distancesum = 0.0
    for i in xrange(1, len(x)):
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        if(dist > 120):
            numberOfHyperSpaceJumps = numberOfHyperSpaceJumps + 1
        v.append(dist)
        distancesum += dist
    return v, distancesum, numberOfHyperSpaceJumps

def acceleration(v):
    """this returns 4 vectors, acceleration at each timestep, acc"""
    accelerationvector = []
    accVector = []
    decVector = []
    timeDecel = 0
    isaccel = False
    isdecel = False
    numOfChanges = 0
    for i in xrange(1, len(v)):
        acc = v[i] - v[i-1]
        accelerationvector.append(acc)
        if(acc > 0):
           isaccel = True
           if(isdecel == True):
                isdecel = False
                numOfChanges = numOfChanges + 1
           accVector.append(acc)
        if(acc < 0):
           timeDecel = timeDecel + 1
           isdecel = True
           if(isaccel == True):
                isaccel = False
                numOfChanges = numOfChanges + 1
           decVector.append(acc)
    return accelerationvector, accVector, decVector, numOfChanges

def jerk(accVect):
    """this returns 3 vectors jerks at each timestep, negative jerks, and positive jerks"""
    jerk = []
    posJerks = []
    negJerks = []
    for i in xrange(1, len(accVect)):
        magOfJerk = accVect[i] - accVect[i-1]
        jerk.append(magOfJerk)
        if(magOfJerk < 0):
            posJerks.append(magOfJerk)
        if(magOfJerk > 0):
            negJerks.append(magOfJerk)
    return jerk, negJerks, posJerks

def calculateDotProduct(xn, yn):
    theta = []
    xn = numpy.array(xn)
    yn = numpy.array(yn)
    for a in range(8, len(xn)):
        firstchangeInX = xn[a-4] - xn[a-8]
        firstchangeInY = yn[a-4] - yn[a-8]
        secondChangeInX = xn[a] - xn[a-4]
        secondChangeInY = yn[a] - yn[a-4]
        dotProduct = firstchangeInX*secondChangeInX + firstchangeInY*secondChangeInY
        magOfFirst = numpy.sqrt((firstchangeInX**2) + (firstchangeInY**2))
        magOfSec = numpy.sqrt(secondChangeInX**2 + secondChangeInY**2)
        if(magOfSec*magOfFirst != 0):
            theta.append(dotProduct/(magOfFirst*magOfSec))
    return theta


def rotational(theta):
    # http://en.wikipedia.org/wiki/Rotation_matrix
    # Beyond rotation matrix, fliping, scaling, shear can be combined into a single affine transform
    # http://en.wikipedia.org/wiki/Affine_transformation#mediaviewer/File:2D_affine_transformation_matrix.svg
    return numpy.array([[numpy.cos(theta),-numpy.sin(theta)],[numpy.sin(theta),numpy.cos(theta)]])

def flip(x):
        # flip a trip if more that half of coordinates have y axis value above 0
        if numpy.sign(x[:,1]).sum() > 0:
            x = x.dot(numpy.array([[1,0],[0,-1]]))

        return x

def rotate_trip(trip):
        # take last element
        a=trip.iloc[-1]
        # get the degree to rotate
        w0=numpy.arctan2(a.y,a.x) # from origin to last element angle
        # rotate using the rotational: equivalent to rotational(-w0).dot(trip.T).T
        return numpy.array(trip.dot(rotational(w0)))


def kalman(x, y, diff):
    """this is to implement a kalman filter"""
    velX = numpy.diff(x,1)
    velY = numpy.diff(y,1)

    newXVel = numpy.add(x[1:], velX)
    newXVel = numpy.add(newXVel[:-1]*0.3, x[2:]*0.7)
    newXVel = numpy.append(x[0:2], newXVel)

    newYVel = numpy.add(x[1:], velY)
    newYVel = numpy.add(newYVel[:-1]*0.3, y[2:]*0.7)
    newYVel = numpy.append(x[0:2], newYVel)

    return newXVel, newYVel, velX, velY, diff


def preProcessing(t1):
    """this is to preprocess the input using a kalman filter

        kalman filters are useful to clean gps data.

    """
    orig = len(t1)
    t1Unique = pd.DataFrame(t1)
    t1Unique =  t1Unique.loc[t1.x.shift() != t1.x]
    difference = abs(len(t1Unique) - len(t1))
    t1r=flip(rotate_trip(t1))
    x = numpy.array(t1r[:,0])
    y = numpy.array(t1r[:,1])
    return kalman(x, y, difference)

def curvature_spline(x, y):
    """
    this function gets the curvature of the road.
    """

    t = numpy.arange(x.shape[0])
    fx = UnivariateSpline(t, x, k=4)
    fy = UnivariateSpline(t, y, k=4)
    xDer = fx.derivative(1)(t)
    xDerDer = fx.derivative(2)(t)
    yDer = fy.derivative(1)(t)
    yDerDer = fy.derivative(2)(t)
    curvature = (xDer * yDerDer - yDer * xDerDer) / numpy.power(xDer ** 2 + yDer ** 2, 3/2)
    return curvature





def how_far_by_percentage_of_trip(yn):
    """we want to know how far we went by a certain percentile of the trip"""
    ynReal25 = yn[int(len(yn) * 0.25)]
    ynReal50 = yn[int(len(yn) * 0.5)]
    ynReal75 = yn[int(len(yn) * 0.75)]
    ynReal80 = yn[int(len(yn) * 0.8)]
    ynReal100 = yn[int(len(yn) * 1) - 1]
    return ynReal100, ynReal25, ynReal50, ynReal75, ynReal80


def distance_covered(xnReal100, xnReal25, xnReal50, xnReal75, ynReal100, ynReal25, ynReal50, ynReal75):
    """
    how much distance did we cover by a percentage of the trip.
    """
    distanceCoveredFirst25 = distance(0, 0, xnReal25, ynReal25)
    distanceCoveredFirst50_25 = distance(xnReal25, ynReal25, xnReal50, ynReal50)
    distanceCoveredFirst50_75 = distance(xnReal50, ynReal50, xnReal75, ynReal75)
    distanceCoveredFirst75_100 = distance(xnReal75, ynReal75, xnReal100, ynReal100)
    return distanceCoveredFirst25, distanceCoveredFirst50_25, distanceCoveredFirst50_75, distanceCoveredFirst75_100


def speed_features(v):
    """
    list of a variety of speed features.
    """
    speedingAbove20 = 0
    speedingAbove15Below20 = 0
    speedingAbove10Below15 = 0
    speedingBelow_3 = 0
    for speeding in v:
        if (speeding > 20):
            speedingAbove20 += 1
        if (speeding > 15 and speeding < 20):
            speedingAbove15Below20 += 1
        if (speeding > 10 and speeding < 15):
            speedingAbove10Below15 += 1
        if (speeding < 0.3):
            speedingBelow_3 += 1
    speed99 = numpy.percentile(v, 99)
    speed90 = numpy.percentile(v, 90)
    speed70 = numpy.percentile(v, 70)
    speed30 = numpy.percentile(v, 30)
    gpsLag = 70 / speed70
    return gpsLag, speed30, speed70, speed90, speed99, speedingAbove10Below15, speedingAbove15Below20, speedingAbove20, speedingBelow_3


def geography_of_plane(yn):
    farthestyPos = numpy.max(yn)
    farthestyNeg = numpy.min(yn)
    farthesty = max(abs(farthestyPos), abs(farthestyNeg))
    return farthesty, farthestyNeg, farthestyPos


def accelerationOverCertainValues(accelerationvector):
    accelAbove2 = 0
    accelBelow2Above0 = 0
    accelAbove_2Below0 = 0
    accelBelow_2 = 0
    accelAbove_5 = 0
    constantMax = -10000
    constant = 0
    constantCount = 0
    goingConstant = False
    for accel in accelerationvector:
        if (accel > 2):
            accelAbove2 += 1
            if (goingConstant):
                goingConstant = False
                if (constantMax < constantCount):
                    constantMax = constantCount
        elif (accel < 2 and accel > 0):
            accelBelow2Above0 += 1
            if (goingConstant):
                goingConstant = False
                if (constantMax < constantCount):
                    constantMax = constantCount
        elif (accel > -2 and accel < 0):
            accelAbove_2Below0 += 1
            if (goingConstant):
                goingConstant = False
                if (constantMax < constantCount):
                    constantMax = constantCount
        elif (accel < -2):
            accelBelow_2 += 1
            if (goingConstant):
                goingConstant = False
                if (constantMax < constantCount):
                    constantMax = constantCount
        elif (accel < -5):
            accelAbove_5 += 1
            if (goingConstant):
                goingConstant = False
                if (constantMax < constantCount):
                    constantMax = constantCount
        if (abs(accel) < 0.5):
            constant += 1
            if (goingConstant):
                constantCount += 1
            else:
                goingConstant = True
                constantCount = 0
    return accelAbove2, accelAbove_2Below0, accelAbove_5, accelBelow2Above0, accelBelow_2, constant, constantMax


def theta_percentiles(maxx, maxy, theta):
    tangentActual = numpy.abs(maxx) / numpy.abs(maxy)
    theta98 = numpy.percentile(theta, 98)
    theta90 = numpy.percentile(theta, 90)
    theta75 = numpy.percentile(theta, 75)
    theta50 = numpy.percentile(theta, 50)
    theta25 = numpy.percentile(theta, 25)
    theta2 = numpy.percentile(theta, 2)
    return tangentActual, theta2, theta25, theta50, theta75, theta90, theta98


def get_major_roads(lengthOfEveryStraightSegments, lengthOfEveryTurnSegment):
    if (len(lengthOfEveryStraightSegments) > 0):
        straightLine1 = lengthOfEveryStraightSegments[-1]
    else:
        straightLine1 = -1
    if (len(lengthOfEveryStraightSegments) > 1):
        straightLine2 = lengthOfEveryStraightSegments[-2]
    else:
        straightLine2 = -1
    if (len(lengthOfEveryTurnSegment) > 0):
        curveLine1 = lengthOfEveryTurnSegment[-1]
    else:
        curveLine1 = -1
    return curveLine1, straightLine1, straightLine2


def length_of_straight_roads(theta):
    epsilon = .05
    lengthOfEveryStraightSegments = []
    lengthOfEveryTurnSegment = []
    count = 0
    turn = 0
    turning = False
    for l in numpy.abs(theta):
        if l < epsilon:
            turning = True
            if (count > 2):
                lengthOfEveryStraightSegments.append(count)
                count = 0
                turn = 0
            turn = turn + 1
        else:
            if turning == True:
                turning = False
                if (turn > 2):
                    lengthOfEveryTurnSegment.append(turn)
                    turn = 0
                    count = 1
            count = count + 1
    return lengthOfEveryStraightSegments, lengthOfEveryTurnSegment


def angle_features(theta):
    angleBelow0 = 0
    anglAbove0 = 0
    anglAbove10 = 0
    anglAbove0_5Below10 = 0
    anglAbove0Below0_5 = 0
    anglAboveNeg0_5Below0 = 0
    anglAboveNeg1Below0_5 = 0
    anglBelowNeg1 = 0
    anglAboveAbs0_5 = 0
    anglBelowAbs0_5 = 0
    for angl in theta:
        if (angl > 0.1):
            anglAbove10 += 1
        if (angl > 0):
            anglAbove0 += 1
        if (angl < 0):
            angleBelow0 += 1
        if (abs(angl) > 0.05):
            anglAboveAbs0_5 += 1
        if (abs(angl) < 0.05):
            anglBelowAbs0_5 += 1
        if (angl > 0.05 and angl < 0.1):
            anglAbove0_5Below10 += 1
        if (angl > 0 and angl < 0.05):
            anglAbove0Below0_5 += 1
        if (angl > -0.05 and angl < 0):
            anglAboveNeg0_5Below0 += 1
        if (angl > -.1 and angl < 0.5):
            anglAboveNeg1Below0_5 += 1
        if (angl < -0.1):
            anglBelowNeg1 += 1

    return anglAbove0, anglAbove0Below0_5, anglAbove0_5Below10, anglAbove10, anglAboveAbs0_5, anglAboveNeg0_5Below0, anglAboveNeg1Below0_5, anglBelowAbs0_5, anglBelowNeg1, angleBelow0


def acceleration_at_angles(accelerationvector, theta):
    accAtAngles = zip(accelerationvector, theta)
    accAtAngles.sort(key=lambda accAtAngles: accAtAngles[1])
    accatAngleBelow_10 = []
    accatAngleBelow_5Above_10 = []
    accatAngleBelow_1Above_5 = []
    accatAngleBelow1Above_1 = []
    accatAngleAbove1Below5 = []
    accatAngleAbove5 = []
    for accAngl in accAtAngles:
        if (accAngl[1] < -10):
            accatAngleBelow_10.append(accAngl[0])
        if (accAngl[1] < -5 and accAngl[1] > -10):
            accatAngleBelow_5Above_10.append(accAngl[0])
        if (accAngl[1] < -1 and accAngl[1] > -5):
            accatAngleBelow_1Above_5.append(accAngl[0])
        if (accAngl[1] < 1 and accAngl[1] > -1):
            accatAngleBelow1Above_1.append(accAngl[0])
        if (accAngl[1] < 5 and accAngl[1] > 1):
            accatAngleAbove1Below5.append(accAngl[0])
        if (accAngl[1] > 5):
            accatAngleAbove5.append(accAngl[0])
    return accatAngleAbove1Below5, accatAngleAbove5, accatAngleBelow1Above_1, accatAngleBelow_10, accatAngleBelow_1Above_5, accatAngleBelow_5Above_10


def avg_velocity_at_angles(velatAngleAbove1Below5, velatAngleAbove5, velatAngleBelow1Above_1, velatAngleBelow_10,
                           velatAngleBelow_1Above_5, velatAngleBelow_5Above_10):
    avgvelatAngleBelow_10 = 0
    avgvelatAngleBelow_5Above_10 = 0
    avgvelatAngleBelow_1Above_5 = 0
    avgvelatAngleBelow1Above_1 = 0
    avgvelatAngleAbove1Below5 = 0
    avgvelatAngleAbove5 = 0
    if (len(velatAngleBelow_10) > 0):
        avgvelatAngleBelow_10 = numpy.average(velatAngleBelow_10)
    if (len(velatAngleBelow_5Above_10) > 0):
        avgvelatAngleBelow_5Above_10 = numpy.average(velatAngleBelow_5Above_10)
    if (len(velatAngleBelow_1Above_5) > 0):
        avgvelatAngleBelow_1Above_5 = numpy.average(velatAngleBelow_1Above_5)
    if (len(velatAngleBelow1Above_1) > 0):
        avgvelatAngleBelow1Above_1 = numpy.average(velatAngleBelow1Above_1)
    if (len(velatAngleAbove1Below5) > 0):
        avgvelatAngleAbove1Below5 = numpy.average(velatAngleAbove1Below5)
    if (len(velatAngleAbove5) > 0):
        avgvelatAngleAbove5 = numpy.average(velatAngleAbove5)
    return avgvelatAngleAbove1Below5, avgvelatAngleAbove5, avgvelatAngleBelow1Above_1, avgvelatAngleBelow_10, avgvelatAngleBelow_1Above_5, avgvelatAngleBelow_5Above_10


def curvature_at_velocities(theta, v):
    VelAtAngles = zip(v, theta)
    VelAtAngles.sort(key=lambda VelAtAngles: VelAtAngles[1])
    velatAngleBelow_10 = []
    velatAngleBelow_5Above_10 = []
    velatAngleBelow_1Above_5 = []
    velatAngleBelow1Above_1 = []
    velatAngleAbove1Below5 = []
    velatAngleAbove5 = []
    for velAngl in VelAtAngles:
        if (velAngl[1] < -10):
            velatAngleBelow_10.append(velAngl[0])
        if (velAngl[1] < -5 and velAngl[1] > -10):
            velatAngleBelow_5Above_10.append(velAngl[0])
        if (velAngl[1] < -1 and velAngl[1] > -5):
            velatAngleBelow_1Above_5.append(velAngl[0])
        if (velAngl[1] < 1 and velAngl[1] > -1):
            velatAngleBelow1Above_1.append(velAngl[0])
        if (velAngl[1] < 5 and velAngl[1] > 1):
            velatAngleAbove1Below5.append(velAngl[0])
        if (velAngl[1] > 5):
            velatAngleAbove5.append(velAngl[0])
    return velatAngleAbove1Below5, velatAngleAbove5, velatAngleBelow1Above_1, velatAngleBelow_10, velatAngleBelow_1Above_5, velatAngleBelow_5Above_10


def tangents(farthestxNeg, farthestxPos, farthestyNeg, farthestyPos, xnReal100, xnReal25, xnReal50, xnReal75, xnReal80,
             ynReal100, ynReal25, ynReal50, ynReal75, ynReal80):
    if (numpy.abs(farthestyPos) > numpy.abs(farthestyNeg)):
        maxy = farthestyPos
    else:
        maxy = farthestyNeg
    if (numpy.abs(farthestxPos) > numpy.abs(farthestxNeg)):
        maxx = farthestxPos
    else:
        maxx = farthestxNeg
    if (maxy == 0.0):
        tangent = 0
    else:
        tangent = maxx / maxy
    if ((maxy - ynReal50) == 0.0):
        tangent50100 = 0
    else:
        tangent50100 = (maxx - xnReal50) / (maxy - ynReal50)
    if (ynReal50 == 0.0):
        tangent0_50 = 0
    else:
        tangent0_50 = xnReal50 / ynReal50
    if (ynReal25 == 0.0):
        tangent0_25 = 0
    else:
        tangent0_25 = xnReal25 / ynReal25
    if ((ynReal50 - ynReal25) == 0.0):
        tangent20_50 = 0
    else:
        tangent20_50 = (xnReal50 - xnReal25) / (ynReal50 - ynReal25)
    if ((ynReal80 - ynReal50) == 0.0):
        tangent50_80 = 0
    else:
        tangent50_80 = (xnReal80 - xnReal50) / (ynReal80 - ynReal50)
    if ((ynReal100 - ynReal75) == 0.0):
        tangent70_100 = 0
    else:
        tangent70_100 = (xnReal100 - xnReal75) / (ynReal100 - ynReal75)

    return maxx, maxy, tangent, tangent0_25, tangent0_50, tangent20_50, tangent50100, tangent50_80, tangent70_100


def percentiles_of_jerk(vVelTupleSortedByJerk):
    averageVelFor20Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[0:20]])
    averageVelFor40Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[20:40]])
    averageVelFor60Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[40:60]])
    averageVelFor80Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[60:80]])
    averageVelFor100Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[80:100]])
    averageVelFor120Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[100:120]])
    averageVelFor140Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[120:140]])
    averageVelFor200Jerk = numpy.average([x for x, _ in vVelTupleSortedByJerk[180:200]])
    return averageVelFor100Jerk, averageVelFor120Jerk, averageVelFor140Jerk, averageVelFor200Jerk, averageVelFor20Jerk, averageVelFor40Jerk, averageVelFor60Jerk, averageVelFor80Jerk


def acceleration_sorted_by_jerk(accelerationvector, jerk, v):
    vAccTupleSortedByJerk = zip(accelerationvector, jerk)
    vVelTupleSortedByJerk = zip(v, jerk)
    vAccTupleSortedByJerk.sort(key=lambda AccJerkTuple: AccJerkTuple[1])
    vVelTupleSortedByJerk.sort(key=lambda VelJerkTuple: VelJerkTuple[1])
    return vVelTupleSortedByJerk


def statistics_of_jerk(jerk, negjerk, posjerk):
    averageOfPositiveJerks = numpy.average(posjerk)
    averageOfNegativeJerks = numpy.average(negjerk)
    rangeOfJerk = abs(numpy.percentile(jerk, 98) - numpy.percentile(jerk, 2))
    jerk90 = numpy.percentile(jerk, 90)
    posjerk30 = numpy.percentile(posjerk, 30)
    posjerk10 = numpy.percentile(posjerk, 10)
    negjerk90 = numpy.percentile(negjerk, 90)
    negjerk70 = numpy.percentile(negjerk, 70)
    negjerk30 = numpy.percentile(negjerk, 30)
    negjerk10 = numpy.percentile(negjerk, 10)
    return averageOfNegativeJerks, averageOfPositiveJerks, jerk90, negjerk10, negjerk30, negjerk70, negjerk90, posjerk10, posjerk30, rangeOfJerk