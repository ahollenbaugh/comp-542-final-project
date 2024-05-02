from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def calculate_f1_scores_on_subsets(X, y, j, test_size):
  max_score = -1;
  best_feature_subset = []
  for i in range(1, j+1):  # Iterate through 1 to j features
    for feature_subset in combinations(range(j), i):  # Generate all "j choose i" subsets
      # print(f"feature_subset = {list(feature_subset)}")
      feature_subset = list(feature_subset)
      X_subset = X[:, feature_subset]  # Select features based on subset indices
      # print(f"X_subset = {X_subset}")
      if not X_subset.any():
          continue
      X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=test_size, random_state=42)

      # Train model, make predictions, and calculate F1 score
      model = RandomForestClassifier()  # Replace with your model
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      f1 = f1_score(y_test, y_pred, average='macro')  # Macro average for multiclass
      if f1 > max_score:
          max_score = f1
          best_feature_subset = feature_subset

  return best_feature_subset
