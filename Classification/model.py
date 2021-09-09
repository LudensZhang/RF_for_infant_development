import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,roc_auc_score


def build_forest(X, y):
    rfc = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = rfc.fit(X_train, y_train)
    print('RFC score', rfc.score(X_test, y_test))
    
    probability=rfc.predict_proba(X_test)
    print(y_test)
    fpr,tpr,thresholds = roc_curve(y_test,probability[:,1], pos_label=1)
    print('AUC=', auc(fpr, tpr))
    
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = clf.fit(X_train, y_train)
    print('DTC score', clf.score(X_test, y_test))
       
    


if __name__ == '__main__':
    C_abundance = pd.read_csv('Dataset/C-abundance.csv')
    C_abundance.set_index(C_abundance.columns[0], inplace=True)
    C_abundance = pd.DataFrame(C_abundance.values.T, columns = C_abundance.index, index = C_abundance.columns)
    C_metadata = pd.read_csv('Dataset/C-metadata.csv')
    
    V_abundance = pd.read_csv('Dataset/V-abundance.csv')
    V_abundance.set_index(V_abundance.columns[0], inplace=True)
    V_abundance = pd.DataFrame(V_abundance.values.T, columns = V_abundance.index, index = V_abundance.columns)
    V_metadata = pd.read_csv('Dataset/V-metadata.csv')
    
    C_X, C_y = np.array(C_abundance.iloc[:, 1:]), C_metadata['Env']
    C_features = C_abundance.columns[1:]
    
    V_X, V_y = np.array(V_abundance.iloc[:, 1:]), V_metadata['Env']
    V_features = V_abundance.columns[1:]

    build_forest(C_X, C_y)
    # build_forest(V_X, V_y)
    
    
    
    
