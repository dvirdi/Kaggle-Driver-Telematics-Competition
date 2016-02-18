from datetime import datetime
import os

__author__ = 'dvirdi'


def comma_seperated_to_vector(line):
    return line.rstrip('\n').rstrip(']').split(",")[1:len(line)]


def get_feature_file(CSV_NAME):
    """get file with features"""
    submissionPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))+"/../data/submission_data"
    featurePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))+"/../data/intermediate_data/"
    featuresFile = featurePath + CSV_NAME
    return submissionPath, featuresFile


def create_submission_file(submission_path):
    """this method creates a new submission_file"""
    return os.path.join(submission_path, "Submission_3_12.csv".format(datetime.now().strftime("%H_%M_%B_%d_%Y")))

