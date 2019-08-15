import numpy as np
import pandas as pd
import pickle

df = [['BRCA1', 'BRCA2', 'CDH1', 'PTEN', 'TP53', 'ATM', 'BARD1', 'CHEK2', 'PALB2', 'RAD51D']]
ps = []

patients = open("/pghbio/cure/tfrisby/cure/expB_data/master_patient_list.txt","r")
for pat in patients:
    try:
        pat = pat.strip('\n')
        data = pd.read_csv("/pghbio/cure/tfrisby/cure/expB_data/salmon_data/"+pat+"/genequant.sf",sep='\s+',header=None,index_col=0)
        vals = data.iloc[:,-1].values
        ps.append(pat)
        df.append(vals)
    except:
        print(pat)
d = pd.DataFrame(df[1:],index=ps,columns=df[0])
print(d)
d.to_pickle("salmon_cancer.pkl")