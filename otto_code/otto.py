import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
df_train = pd.read_csv('/Users/pooja/Development/datascience/repos/otto_dataset/train.csv')
X_train = pd.DataFrame(df_train.iloc[:,1:94])
y_train = df_train.iloc[:,94]
categories = df_train['target'].unique()
df_test = pd.read_csv('/Users/pooja/Development/datascience/repos/otto_dataset/test.csv')

X_test = pd.DataFrame(df_test.iloc[:,1:94])
X_test_ids = df_test.iloc[:,0]
#print "\nRANDOM FOREST\n"
# see what happens as we bump up the number of estimators
random_forest_clf = RandomForestClassifier(n_estimators=5)
random_forest_clf.fit(X_train, y_train)
clf_probs = random_forest_clf.predict_proba(X_test)
df_output = pd.DataFrame(clf_probs)
#df_output.shape
#scores = cross_val_score(random_forest_clf, features, target, cv=5)
#print "Mean: {}".format(scores.mean())
#print "Std Dev: {}".format(np.std(scores))
df_output.columns = categories
df_output = pd.concat([X_test_ids,df_output], axis=1)
df_output.to_csv('/Users/pooja/Development/datascience/repos/otto_dataset/out.csv', index=False)
