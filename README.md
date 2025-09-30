# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```python
import pandas as pd
df = pd.read_csv('income(1) (1).csv')
df
```

<img width="926" height="552" alt="image" src="https://github.com/user-attachments/assets/4853588f-8b19-45a9-8c4a-a1ff3314db70" />

```python
df.shape
```

<img width="97" height="25" alt="image" src="https://github.com/user-attachments/assets/d9762cc3-11ef-4192-8960-e71cfd9f13b1" />

```python
from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
X = df_encoded.drop('SalStat', axis=1)
y = df_encoded['SalStat']

print(X)
```
<img width="611" height="432" alt="image" src="https://github.com/user-attachments/assets/3ad50c14-5a96-4323-bbb9-33b9ab19507a" />

```python
print(y)
```

<img width="338" height="197" alt="image" src="https://github.com/user-attachments/assets/39d58ec6-daf6-4bb2-88cd-b36bfd9c2fe3" />

```python
from sklearn.feature_selection import SelectKBest, chi2
chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X, y)

selected_features_chi2 = X.columns[chi2_selector.get_support()]
print("Selected Features (Chi-Square):", list(selected_features_chi2))

mi_score = pd.Series(chi2_selector.scores_, index=X.columns)
print(mi_score.sort_values(ascending = False))
```

<img width="737" height="222" alt="image" src="https://github.com/user-attachments/assets/ca6f55b2-ad1b-4ad0-b169-3e326edc0ed5" />

```python
from sklearn.feature_selection import f_classif
anova_selector = SelectKBest(f_classif, k=5)
anova_selector.fit(X, y)

selected_features_anova = X.columns[anova_selector.get_support()]
print("Selected Features (ANOVA F-test):", list(selected_features_anova))

mi_score = pd.Series(anova_selector.scores_, index=X.columns)
print(mi_score.sort_values(ascending = False))
```
<img width="720" height="232" alt="image" src="https://github.com/user-attachments/assets/18b6d16d-6f37-4f4a-8ca5-fa5b3eaca05a" />

```python

from sklearn.feature_selection import mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=5)
mi_selector.fit(X, y)

selected_features_mi = X.columns[mi_selector.get_support()]
print("Selected Features (Mutual Info):", list(selected_features_mi))

mi_score = pd.Series(mi_selector.scores_, index=X.columns)
print("\nMutual Information Scores:\n", mi_score.sort_values(ascending = False))
```
<img width="720" height="267" alt="image" src="https://github.com/user-attachments/assets/9763384e-4751-41e5-85ad-9942a3c7f9f4" />


```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

selected_features_rfe = X.columns[rfe.support_]
print("Selected Features (RFE):", list(selected_features_rfe))
```
<img width="635" height="31" alt="image" src="https://github.com/user-attachments/assets/94cf7f2b-bda6-4892-81e2-1d5b24c04463" />

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS

model = LogisticRegression(max_iter=100)
sfs = SFS(model, n_features_to_select=5)
sfs.fit(X, y)

selected_features_sfs = X.columns[sfs.support_]
print("Selected Features (SFS):", list(selected_features_sfs))
```

<img width="716" height="37" alt="image" src="https://github.com/user-attachments/assets/a5e530ef-187a-4b95-a6ba-017d33c41053" />

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print("Top 5 features (Random Forest Importance):", list(selected_features_rf))
```
<img width="753" height="42" alt="Screenshot 2025-09-30 144351" src="https://github.com/user-attachments/assets/f350faa7-27b4-4ecf-a733-a90173c1cc8c" />

```python
from sklearn.linear_model import LassoCV
import numpy as np

lasso = LassoCV(cv=5).fit(X, y)
importance = np.abs(lasso.coef_)

selected_features_lasso = X.columns[importance > 0]
print("Selected Features (Lasso):", list(selected_features_lasso))

```
<img width="586" height="22" alt="image" src="https://github.com/user-attachments/assets/a8e1a1aa-632b-4ed7-8fc1-aa36ea84157a" />

```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv('income(1) (1).csv')
le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
X = df_encoded.drop('SalStat', axis=1)
y = df_encoded['SalStat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
<img width="471" height="276" alt="image" src="https://github.com/user-attachments/assets/9f5b4b63-e20d-4ab1-a230-e65682a71d7e" />




# RESULT:
     
Thus, the Feature selection and Feature scaling has been used on the given dataset and executed successfully.
