import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


iris = load_iris()
score_log = cross_val_score(LogisticRegression(), iris.data, iris.target)
print(np.average(score_log), "Logistic Regression")
score_svm = cross_val_score(SVC(),iris.data,iris.target)
print(np.average(score_svm), "SVM")
score_rand = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data,iris.target)
print(np.average(score_rand), "Random Forest")