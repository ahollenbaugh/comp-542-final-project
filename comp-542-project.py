import pandas
from sklearn.feature_selection import r_regression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import f1_combinations as f1

# Read in and process the dataset file:
# dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\AndroidAdware2017\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\Data Sets\\diabetes.csv", sep=',')
dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

# Grab feature columns and class label column:
X = dataset.iloc[:, :-1].values # observations/training examples without class label (final column)
y = dataset.iloc[:, -1].values # class label for each observation

# Calculate the Pearson correlation coefficient between features and class:
pearson_correlation_coefficients = r_regression(X, y)

# Create DataFrame and sort by the 'Correlation' column in descending order: 
features = dataset.columns[:-1]
correlation_df = pandas.DataFrame({'Feature': features, 'Correlation': pearson_correlation_coefficients})
sorted_df = correlation_df.sort_values(by='Correlation', ascending=False)
print(sorted_df)

# Generate different sets/combinations of features and calculate the F1 score:
j = 20
test_size = 0.25
feature_subset_highest_f1 = f1.calculate_f1_scores_on_subsets(X, y, j, test_size)
print(f"This set of features has the highest f1 score: {feature_subset_highest_f1.columns[:-1]}")

# Get information gain for each feature:
clf = DecisionTreeClassifier()
clf.fit(X, y)
IG = clf.feature_importances_
print(IG)

# Scale the data:
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

# Cross-validation:
rfc = RandomForestClassifier()
result = cross_validate(rfc, X_scaled, y)
print(result['test_score'])

print("all done!")