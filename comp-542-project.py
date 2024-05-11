import numpy as np
import pandas as pd
import time
from sklearn.feature_selection import r_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import f1_combinations as f1

# Start time for runtime duration calculation
start_time = time.time()

# Read in and process the dataset file:
# dataset = pd.read_csv("C:\\Users\\wjona\\OneDrive\\Documents\\CSUN\\Comp542\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
dataset = pd.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\AndroidAdware2017\\TotalFeatures-ISCXFlowMeter.csv", sep=',')

dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

# Grab feature columns and class label column:
X = dataset.iloc[:, :-1].values # observations/training examples without class label (final column)
y = dataset.iloc[:, -1].values # class label for each observation

'''
In order to use binary classification, we need to map the class label to
some binary value.

In this case, we're mapping asware and GeneralMalware to True,
and benign to False.
'''
# Convert y to Series for replace
y_temp = pd.Series(y)
y_temp = y_temp.replace({'benign': 0, 'asware': 1, 'GeneralMalware': 1})

# Convert back to NumPy array
y = y_temp.to_numpy()
print(y)

# Calculate the Pearson correlation coefficient between features and class:
pearson_correlation_coefficients = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]

print("Pearson correlation coefficients:", pearson_correlation_coefficients[0]) # print coefficients


# Create DataFrame and sort by the 'Correlation' column in descending order: 
features = dataset.columns[:-1]
correlation_df = pd.DataFrame({'Feature': features, 'Correlation': pearson_correlation_coefficients})
sorted_df = correlation_df.sort_values(by='Correlation', ascending=False)

print("\nSummary of the dataset:")
print(sorted_df.info())
print("\nSorted DataFrame:")
print(sorted_df)

# ********** NEW CHECKPOINT ************

# Generate different sets/combinations of features and calculate the F1 score:
j = 4 # CAREFUL! Make sure j doesn't exceed the number of features!
# 1. Grab the first j feature names from sorted_df and put them in a list.
top_j_feature_names = list(sorted_df.iloc[0:j+1, 0])
# print(top_j_feature_names)
# 2. Create a copy of dataset only containing said columns.
dataset_top_j_features = dataset[top_j_feature_names]
# 3. Make a new version of X.
X_top_j = dataset_top_j_features.iloc[:, :-1].values
# 4. Run f1 function on copy of dataset.
test_size = 0.25
feature_subset_highest_f1 = f1.calculate_f1_scores_on_subsets(X_top_j, y, j, test_size)
#print(f"This set of features has the highest f1 score: {dataset_top_j_features.columns[feature_subset_highest_f1]}")

# ********** NEW CHECKPOINT ************

# Get information gain for each feature:
clf = DecisionTreeClassifier()
clf.fit(X, y)
IG = clf.feature_importances_
# print(f"information gain results: {IG}")

# ********** NEW CHECKPOINT ************

# 1. Create a dataframe containing features and their IG values.
info_gain_df = pd.DataFrame({'Feature': features, 'IG': IG})
# 2. Sort dataframe by IG in descending order.
sorted_info_gain_df = info_gain_df.sort_values(by='IG', ascending=False)

# ********** NEW CHECKPOINT ************ -- Runtime = 133 seconds up to this point

# 3. Generate different sets/combinations of features and calculate the F1 score:
k = 4
top_k_feature_names = list(sorted_info_gain_df.iloc[0:k+1, 0])
dataset_top_k_features = dataset[top_k_feature_names]
X_top_k = dataset_top_k_features.iloc[:, :-1].values
feature_subset_highest_f1_IG = f1.calculate_f1_scores_on_subsets(X_top_k, y, k, test_size)
# print(f"This set of features has the highest f1 score: {dataset_top_k_features.columns[feature_subset_highest_f1_IG]}")

# ********** NEW CHECKPOINT ************ -- Runtime = 259 seconds up to this point

# Take the union of the two sets of features.
selected_features = set(feature_subset_highest_f1).union(set(feature_subset_highest_f1_IG)) # Stuck here
# print(f"selected features: {selected_features}")

X = X[:, list(selected_features)]
print("\nSelected Features:")
print(X)

# ********** NEW CHECKPOINT ************ -- Runtime = 628 seconds up to this point

# Scale the data:
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print("\nSelected Features after MinMax Scaling:")
print(X_scaled)

# ********** NEW CHECKPOINT ************ -- Runtime = 630 seconds up to this point

# Cross-validation:
rfc = RandomForestClassifier()
result = cross_validate(rfc, X_scaled, y)
print(result['test_score'])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Set random_state for reproducibility

# Train a classifier (e.g., RandomForestClassifier)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rfc.predict(X_test)

# ********** NEW CHECKPOINT ************ -- Runtime = 653 seconds up to this point (without cross-validation)
# ********** NEW CHECKPOINT ************ -- Runtime = 770 seconds up to this point (WITH 5-Fold cross-validation)

# Calculate ROC curve metrics using scikit-learn
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Print AUC score
print(f"AUC score: {roc_auc}")

# Plot the ROC curve (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Diabetes Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



# End time of duration
end_time = time.time()
runtime = end_time - start_time
print("\nRuntime:", runtime, "seconds")

print("All done!")