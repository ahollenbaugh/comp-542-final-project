# Read in the dataset file:
import pandas
from sklearn.feature_selection import r_regression
# dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\AndroidAdware2017\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
dataset = pandas.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\Data Sets\\diabetes.csv", sep=',')
dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

# Grab feature columns and class label column:
X = dataset.iloc[:, :-1].values # observations/training examples without class label (final column)
y = dataset.iloc[:, -1].values # class label for each observation

# Calculate the Pearson correlation coefficient between features and class:
pearson_correlation_coefficients = r_regression(X, y)

# print(pearson_correlation_coefficients) # print coefficients

# Assuming feature_names is a list containing your feature names
features = dataset.columns[:-1]
correlation_df = pandas.DataFrame({'Feature': features, 'Correlation': pearson_correlation_coefficients})

# Sort the DataFrame by the 'Correlation' column in descending order 
sorted_df = correlation_df.sort_values(by='Correlation', ascending=False)

# Now sorted_df contains features and coefficients in descending order by correlation

print(sorted_df)
print("all done!")