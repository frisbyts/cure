import pandas as pd
import numpy as np
import re
import pickle

def label_mask(df,labels,fname):
    mask = np.where(labels.values != "Unknown")[0]
    labels = labels.iloc[mask]
    labels.iloc[np.where(labels.values == "GTE_5YRS")] = 1 
    labels.iloc[np.where(labels.values == "LT_5YRS")] = 0 
    labels.to_pickle(fname)
    return df.iloc[mask]  

def format_data(clinical_spreadsheet,imaging_spreadsheet,salmon_data):
    #load clinical data
    data = pd.read_csv(clinical_spreadsheet,index_col='C1')
    labels = data.iloc[:,-3:]
    labels.columns = ["Survive5","Survive10","DiseaseFree5"]
    labels = labels.drop(labels.index[[0,1]])   
 
    known_at_diagnosis = data.iloc[1,:] == "1" 
    df = data[known_at_diagnosis.index[known_at_diagnosis]]
    df = df.drop(df.index[[0,1]])
    #obtain desired clinical features 
    df = df.loc[:,["C13","C24","C25","C26","R40","R41","R42","R1","R5"]]

    df_surv5 = label_mask(df,labels["Survive5"],"surv5_clinical_labels.pkl")
    df_dfree5 = label_mask(df,labels["DiseaseFree5"],"dfree5_clinical_labels.pkl")

    #load imaging data
    data = pd.read_csv(imaging_spreadsheet,sep='\s+',index_col='ID')
    
    #TCGA-BH-A28P not in imaging data set
    imaging_labels = labels.drop(["TCGA-BH-A28P"],axis=0)
    df_surv5_imaging = label_mask(data,imaging_labels["Survive5"],"surv5_imaging_labels.pkl")
    df_dfree5_imaging = label_mask(data,imaging_labels["DiseaseFree5"],"dfree5_imaging_labels.pkl")    
   
    df_surv5_imaging.to_pickle("LOO_surv5_imaging.pkl")
    df_dfree5_imaging.to_pickle("LOO_dfree5_imaging.pkl")
 
    #load salmon data
    #TCGA-BH-A28P not in salmon data
    df_salmon = pickle.load( open(salmon_data, "rb") )

    surv5_salmon = label_mask(df_salmon,imaging_labels["Survive5"],"surv5_salmon_labels.pkl")
    dfree5_salmon = label_mask(df_salmon,imaging_labels["DiseaseFree5"],"dfree5_salmon_labels.pkl")
    surv5_salmon.to_pickle("LOO_surv5_salmon.pkl")
    dfree5_salmon.to_pickle("LOO_dfree5_salmon.pkl")