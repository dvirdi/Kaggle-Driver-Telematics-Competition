# Kaggle_Driver_Telematics_Competition

#
##Approach:
Take 1000 random trips. This will be your training data of bad examples.
For each folder, the 200 trips will be your training data of good examples.
Run Random Forest, and predict probabilities.
#
Use write_features_to_csv.py to generate csv file of 120 features from all of the drives.
Use feature_generator.py to generate submission file.

##Trip Matching:
Use draw_repeated_trip.py
For all matched trips, average probabilities. If above a certain threshold, set to 1.

##Model Used:
model.py
RandomForestClassifier with n_estimators = 3000.


