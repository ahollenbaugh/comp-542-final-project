import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

import f1_combinations as f1

# Read in and process the dataset file:
dataset = pd.read_csv("C:\\Users\\wjona\\OneDrive\\Documents\\CSUN\\Comp542\\TotalFeatures-ISCXFlowMeter.csv", sep=',')

#dataset = pd.read_csv("C:\\Users\\wjona\\OneDrive\\Documents\\CSUN\\Comp542\\diabetes.csv", sep=',')

# dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\AndroidAdware2017\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
# dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\Data Sets\\diabetes.csv", sep=',')

dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

# Grab feature columns and class label column:
X = dataset.iloc[:, :-1].values # observations/training examples without class label (final column)
y = dataset.iloc[:, -1].values # class label for each observation

# Encode class labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Calculate the Pearson correlation coefficient between features and class:
# pearson_correlation_coefficients = r_regression(X, y)
pearson_correlation_coefficients = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]

print("Pearson correlation coefficients:", pearson_correlation_coefficients[0]) # print coefficients


# Create DataFrame and sort by the 'Correlation' column in descending order: 
features = dataset.columns[:-1]
correlation_df = pd.DataFrame({'Feature': features, 'Correlation': pearson_correlation_coefficients})
sorted_df = correlation_df.sort_values(by='Correlation', ascending=False)

print("\nSorted DataFrame:")
print(sorted_df)

# Generate different sets/combinations of features and calculate the F1 score:
j = 9   # Arbitrary; take the highest j number of features. CAREFUL! j cannot exceed the number of features of the dataset
test_size = 0.25
feature_subset_highest_f1 = f1.calculate_f1_scores_on_subsets(X, y, j, test_size)
print(f"This set of features has the highest f1 score: {feature_subset_highest_f1}")

# Get information gain for each feature:
clf = DecisionTreeClassifier()
clf.fit(X, y)
IG = clf.feature_importances_

print("\nInformation Gain:")
print(IG)

# Scale the data:
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print("\nScaled Data:")
print(X_scaled)

# Cross-validation:
rfc = RandomForestClassifier()
result = cross_validate(rfc, X_scaled, y)

print("\nCross-validation results:")
print(result['test_score'])

print("all done!")