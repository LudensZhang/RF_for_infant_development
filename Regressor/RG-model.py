import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from plotnine import*
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split

def buid_RFR(X, y):
    rfr = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    pd.DataFrame(list(zip(y_test, y_pred)), columns = ['y_test', 'y_pred']).to_csv('predict-result.csv', index=0)
    print(mae)
    # p = (ggplot(aes(x = y_test, y = y_pred))+
    #             geom_point(color = '#66CCFF', alpha = 0.2)+
    #             geom_smooth()+
    #             theme_bw()+
    #             xlim(0,2)+
    #             ylim(0,2)+
    #             xlab('Report Age')+
    #             ylab('Predict Age')+
    #             theme(text = element_text(size = 20)))
    # p.save('plot.png', width = 5, height = 5)


if __name__ == '__main__':
    abundance = pd.read_csv('Dataset/abundance1.csv')
    abundance.set_index(abundance.columns[0], inplace=True)
    abundance = pd.DataFrame(abundance.values.T, columns = abundance.index, index = abundance.columns)
    metadata = pd.read_csv('Dataset/metadata.csv')

    X, y = np.array(abundance), metadata['host_age']
    features = abundance.columns

    buid_RFR(X, y)