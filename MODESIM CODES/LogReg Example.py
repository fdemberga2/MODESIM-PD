import pandas as pd
col_names = ['gmat','gpa','work_Experience','admitted']
pima = pd.read_csv("candidates.csv", header = None, names = col_names)
print(pima.head())
print(pima.tail())
feature_cols = ['gmat','gpa','work_Experience']
X = pima[feature_cols]
Y = pima.admitted
print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter= 1000, penalty= "none")
logreg.fit(X_train, Y_train)

y_pred = logreg.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
print(cnf_matrix)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class_names =[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))

print(logreg.coef_)
print(logreg.intercept_)

import statsmodels.api as sm
X_const = sm.add_constant(X_train)
logit_model = sm.Logit(Y_train, X_const).fit()
print(logit_model)

feature_cols = ['gmat','gpa','work_Experience']
new_patients = {'gmat':[591,740,680,610,710],
               'gpa':[2.0,3.7,3.3,2.3,3.0],
               'work_Experience':[3,4,6,1,5],
               }
df_new_patients = pd.DataFrame(new_patients, columns = ['gmat','gpa','work_Experience'])

y_pred=logreg.predict(df_new_patients)

print (df_new_patients)
print (y_pred)