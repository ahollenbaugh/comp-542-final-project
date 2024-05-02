# Code Skeleton
### IFS Stage
1. ~~Read in the dataset file~~
2. ~~Create a list of values corresponding to each feature, using ```pandas.iloc()```~~
3. ~~Create a list of values corresponding to the class label the same way~~
4. ~~Calculate the Pearson correlation coefficient between each feature and the class label~~
5. ~~Sort the absolute value of the coefficients in descending order~~
6. ~~Choose the top j features from that list. Like, just choose an arbitrary value of j. For 0 <= i < j, create "j choose i" combinations of features.~~
7. ~~For each set C_j, calculate the F1 score~~
8. ~~Select the set C_j* with the highest F1 score~~
9. ~~Calculate the entropy of each feature~~
10. ~~Figure out the chi-squared distribution~~
11. Select the top 20% of features
12. For 1 <= k <= N, create a set C_k containing:
    - k features
    - the class label
13. For each set C_k, calculate the F1 score
14. Select the set C_k* with the highest F1 score
15. Take the union of sets C_j* and C_k*

### Data Scaling Stage
1. ~~Apply MinMaxScaler to the processed dataset (in this case, the data for each feature is scaled to a value between 0 and 1 -- this prevents underflow and overflow when learning from experimental data)~~

### K-Fold Cross Validation Stage
1. ~~Select a model~~
    - ~~general ML algorithm provided by scikit-learn~~
    - ~~random forest (highest prediction accuracy overall)~~
    - ~~decision tree~~
    - ~~k-nearest neighbors classifier~~
    - ~~gradient boosting classifier~~
    - ~~extra tree classifier (shortest learning time)~~
    - ~~bagging classifier~~
2. Create a subset of the dataset, which we'll call D_ifs, where any features deemed irrelevant by IFS are filtered out
3. ~~For 1 <= k <= N, where k now represents the number of folds:~~
    - ~~Select 1/kth of D_ifs~~
    - ~~Train model (binary classification) with 75% of that 1/kth portion of data~~
    - ~~Obtain values of hyperparameters: F1 score (try using their newly proposed F1 scoring method -- obtain F1 score directly while model is training, store value of confusion matrix) and elapsed time~~
    - ~~Test model with the other 25%~~
4. Generate ROC curve to evaluate experimental results