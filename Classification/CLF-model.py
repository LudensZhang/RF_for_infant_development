import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,roc_auc_score


# 构建随机森林
def build_forest(X, y):       
    rfc = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = rfc.fit(X_train, y_train)
    print('RFC score', rfc.score(X_test, y_test))
    y_pred_prob = rfc.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
    print('RFC auc score', auc_score)


# 构建决策树  
def build_DTC(X, y):    
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = clf.fit(X_train, y_train)
    print('DTC score', clf.score(X_test, y_test))
    y_pred_prob = clf.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
    print('DTC auc score', auc_score)
          

if __name__ == '__main__':
    # 数据转置
    C_abundance = pd.read_csv('Dataset/C-abundance.csv')
    C_abundance.set_index(C_abundance.columns[0], inplace=True)
    C_abundance = pd.DataFrame(C_abundance.values.T, columns = C_abundance.index, index = C_abundance.columns)
    C_metadata = pd.read_csv('Dataset/C-metadata.csv')
    
    V_abundance = pd.read_csv('Dataset/V-abundance.csv')
    V_abundance.set_index(V_abundance.columns[0], inplace=True)
    V_abundance = pd.DataFrame(V_abundance.values.T, columns = V_abundance.index, index = V_abundance.columns)
    V_metadata = pd.read_csv('Dataset/V-metadata.csv')
    
    # 分离特征、标签
    C_X, C_y = np.array(C_abundance), C_metadata['Env']
    C_features = C_abundance.columns
    
    V_X, V_y = np.array(V_abundance), V_metadata['Env']
    V_features = V_abundance.columns  
    
    build_forest(C_X, C_y)
    build_DTC(C_X, C_y)
    build_forest(V_X, V_y)
    build_DTC(C_X, C_y)
    
    
    
    
