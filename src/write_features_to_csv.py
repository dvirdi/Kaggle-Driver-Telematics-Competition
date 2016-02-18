import os
from random import seed
from datetime import datetime
import sys

from src import feature_generator
import pandas as pd

__author__ = 'daljeetv'

"""
This file writes features to a csv.
"""

def generateFeatures(folder, f):
    id = int(os.path.basename(f).split(".")[0])
    t1=pd.read_csv(os.path.join(folder, f))
    a = os.path.basename(folder).split(".")[0]
    return feature_generator(t1.x, t1.y, t1,a, id)


def perform_analysis(folder):
    print "Working on {0}".format(folder)
    sys.stdout.flush()
    results = ""
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            results += generateFeatures(folder, f)
    return results

def analysis(foldername, outdir):
    seed(42)
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
    with open(os.path.join(outdir, "3.11.15-featureGeneration.csv".format(submission_id)), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for folder in folders:
            results = perform_analysis(folder)
            for item in results.split("]["):
                writefile.write("%s\n" % item)
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    analysis(os.path.join(MyPath,"..","drivers"), MyPath)