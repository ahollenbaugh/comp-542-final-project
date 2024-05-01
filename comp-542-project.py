# Read in the dataset file:
import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

#dataset = pd.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\AndroidAdware2017\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
dataset = pd.read_csv("C:\\Users\\wjona\\OneDrive\\Documents\\CSUN\\Comp542\\TotalFeatures-ISCXFlowMeter.csv", sep=',')
dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

# Grab feature columns and class label column:
X = dataset.iloc[:, :-1].values # observations/training examples without class label (final column)
y = dataset.iloc[:, -1].values # class label for each observation

# Encode class labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Calculate the Pearson correlation coefficient between features and class:
#pearson_correlation_coefficients = r_regression(X, y)
pearson_correlation_coefficients = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]


print(pearson_correlation_coefficients[0]) # print coefficients

print("all done!")