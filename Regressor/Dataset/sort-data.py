import pandas as pd


if __name__ == '__main__':
    abundance = pd.read_csv('abundance.csv')
    metadata = pd.read_csv('RG-metadata.csv')
    filtered_labels = []
    
    for i in metadata.index:
        if((metadata.loc[i, 'host_age'] > 0) and (metadata.loc[i, 'host_age'] <= 2)):
            filtered_labels.append(i)
    
    filtered_metadata = metadata.loc[filtered_labels]
    filtered_metadata.drop(filtered_metadata.columns[0], axis=1, inplace=True)
    
    filtered_abundance = abundance[filtered_metadata['id']]
    filtered_abundance.insert(0, 'Samples', abundance['Samples'])
    
    filtered_metadata.to_csv('metadata.csv', index=0)
    filtered_abundance.to_csv('abundance1.csv', index=0)
    
    

