import pandas as pd


if __name__ == '__main__':
    abundance = pd.read_csv('CLF-abundance.csv')
    metadata = pd.read_csv('CLF-metadata.csv')
    
    C_index = []
    V_index = []
    
    #按出生方式分类
    for i in metadata.index:
        if ':C' in metadata.loc[i, 'Env']:
            C_index.append(i)
        elif ':V' in metadata.loc[i, 'Env']:
            V_index.append(i)
    
    C_metadata = metadata.loc[C_index]
    C_metadata.columns = metadata.columns
    V_metadata = metadata.loc[V_index]
    V_metadata.columns = metadata.columns
    
    #简化标签
    C_metadata['Env'] = C_metadata['Env'].str.split(':', expand=True)[1]
    V_metadata['Env'] = V_metadata['Env'].str.split(':', expand=True)[1]
    
    #分类丰度表
    C_abundance = abundance[C_metadata['SampleID']]
    C_abundance.insert(0, 'Samples', abundance['Samples'])
    V_abundance = abundance[V_metadata['SampleID']]
    V_abundance.insert(0, 'Samples', abundance['Samples'])
    
    #导出数据
    C_metadata.to_csv('C-metadata.csv', index=0)
    C_abundance.to_csv('C-abundance.csv', index=0)
    V_metadata.to_csv('V-metadata.csv', index=0)
    V_abundance.to_csv('V-abundance.csv', index=0)
