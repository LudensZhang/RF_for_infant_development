import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from plotnine import*

# 构建随机森林
def build_forest(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc = rfc.fit(X_train, y_train)
    y_pred_prob = rfc.predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return(roc_auc)


if __name__ == '__main__':
    C_abundance = pd.read_csv('Dataset/C-abundance.csv')
    C_abundance.set_index(C_abundance.columns[0], inplace=True)
    C_abundance = pd.DataFrame(C_abundance.values.T, columns = C_abundance.index, index = C_abundance.columns)
    C_metadata = pd.read_csv('Dataset/C-metadata.csv')
    
    V_abundance = pd.read_csv('Dataset/V-abundance.csv')
    V_abundance.set_index(V_abundance.columns[0], inplace=True)
    V_abundance = pd.DataFrame(V_abundance.values.T, columns = V_abundance.index, index = V_abundance.columns)
    V_metadata = pd.read_csv('Dataset/V-metadata.csv')
    
    C_X, C_y = np.array(C_abundance), C_metadata['Env']
    C_features = C_abundance.columns
    C_y = label_binarize(C_y, classes=['B', '4M', '12M', '3Y', '5Y'])

    V_X, V_y = np.array(V_abundance), V_metadata['Env']
    V_features = V_abundance.columns
    V_y = label_binarize(V_y, classes=['B', '4M', '12M', '3Y', '5Y'])

    result = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    kf = KFold(n_splits=8)
    for train_index, test_index in kf.split(C_X):
        X_train, X_test = C_X[train_index], C_X[test_index]
        y_train, y_test = C_y[train_index], C_y[test_index]
        result = result.append(build_forest(X_train, y_train, X_test, y_test), ignore_index=1)
    result['mode'] = 'caesarean section'

    for train_index, test_index in kf.split(V_X):
        X_train, X_test = V_X[train_index], V_X[test_index]
        y_train, y_test = V_y[train_index], V_y[test_index]
        result = result.append(build_forest(X_train, y_train, X_test, y_test), ignore_index=1)
    result.fillna('vaginal delivery', inplace = True)
    result.columns = ['NB', '4M', '12M', '3Y', '5Y', 'mode']
    result = result.melt(id_vars='mode')

    p = (ggplot(result, aes(x = 'variable', y = 'value', fill = 'mode'))+
         geom_boxplot()+
         scale_fill_manual(values = ["#DC143C", "#87CEEB"])+
         theme_bw()+
         xlim('NB', '4M', '12M', '3Y', '5Y')+
         ylim(0,1)+
         xlab('')+
         ylab('AUROC')+
         theme(text = element_text(size = 20),
         legend_title = element_blank(),
         legend_position = (0.78, 0.2),
         legend_text = element_text(size=10)))
    print(p)
    p.save('8fold-result.png', height=5, width=8)
    
    
    
    
