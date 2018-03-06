import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')

dia = pd.read_csv("diabetes.csv")

dia.head()

dia.describe()

dia.groupby("Outcome").size()

dia.hist(figsize=(10,8))

dia.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))

column_x = dia.columns[0:len(dia.columns) - 1]

corr = dia[dia.columns].corr()
sns.heatmap(corr, annot = True)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = dia.iloc[:,0:8]
Y = dia.iloc[:,8]
select_top_4 = SelectKBest(score_func=chi2, k = 4)

fit = select_top_4.fit(X,Y)
features = fit.transform(X)

dia.head()

X_features = pd.DataFrame(data = features, columns = ["Glucose","Insulin","BMI","Age"])

X_features.head()

from sklearn.preprocessing import StandardScaler
rescaledX = StandardScaler().fit_transform(X_features)

X = pd.DataFrame(data = rescaledX, columns= X_features.columns)

X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 22, test_size = 0.5)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))


results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean())


ax = sns.boxplot(data=results)
ax.set_xticklabels(names)

lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions = lr.predict(X_test)


#final prediction
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print(accuracy_score(Y_test,predictions))

svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)

print(accuracy_score(Y_test,predictions))

print(classification_report(Y_test,predictions))

conf = confusion_matrix(Y_test,predictions)
plt.show()

label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)
plt.show()




