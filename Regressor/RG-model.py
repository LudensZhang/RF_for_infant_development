import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from plotnine import*
import skmisc

def buid_RFR(X, y):
    rfr = RandomForestRegressor()
    rfr.fit(X, y)
    y_pred = rfr.predict(X)
    p = (ggplot(aes(x = y, y = y_pred))+
                geom_point(color = '#66CCFF', alpha = 0.1)+
                geom_smooth(method = 'lm', span = 1)+
                theme_bw()+
                xlab('Report Age')+
                ylab('Predict Age'))
    p.save('plot.png', width = 4, height = 4)


if __name__ == '__main__':
    abundance = pd.read_csv('Dataset/abundance.csv')
    abundance.set_index(abundance.columns[0], inplace=True)
    abundance = pd.DataFrame(abundance.values.T, columns = abundance.index, index = abundance.columns)
    metadata = pd.read_csv('Dataset/metadata.csv')

    X, y = np.array(abundance), metadata['host_age']
    features = abundance.columns

    buid_RFR(X, y)
<<<<<<< HEAD


=======
    
    
>>>>>>> parent of 4ba19c1 (edit on pycharm)
