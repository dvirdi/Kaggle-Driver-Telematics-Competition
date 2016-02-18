import os
from random import seed
from datetime import datetime
import pandas as pd
import numpy as np
__author__ = 'daljeetv'


def generateData(foldername, folder, f):
    id = int(os.path.basename(f).split(".")[0])
    t1=pd.read_csv(os.path.join(folder, f))
    a = os.path.basename(folder).split(".")[0]
    pureX = t1.x
    pureY = t1.y
    noise = np.random.normal(0, 1.5, len(pureX))
    signalY = pureY + noise
    signalX = pureX + noise


    result = zip(signalX, signalY)
    # pd.data
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(os.path.join(MyPath,"noisyData",folder.split("/")[-1])):
        os.makedirs(os.path.join(MyPath,"noisyData",folder.split("/")[-1]))
    np.savetxt(os.path.join(MyPath,"noisyData",folder.split("/")[-1],f), result, delimiter=",", header="x,y", comments='')

def perform_analysis(foldername, folder):
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            generateData(foldername,folder, f)



def analysis(foldername, MyPath):
    seed(42)
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
    for folder in folders:
        perform_analysis(foldername,folder)


if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    analysis(os.path.join(MyPath,"..","drivers"), MyPath)