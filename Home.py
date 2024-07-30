import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(
    page_title=" DataCo Dashboard",
    page_icon=":bar_chart:",
)

st.title("Welcome to DataCO")
st.sidebar.success("Select a page above.")


'''
A word regarding the dataset selected and used:

A DataSet of Supply Chains used by the company DataCo Global was used for the analysis. Dataset of Supply Chain , which allows the use of Machine Learning Algorithms and R Software.
Areas of important registered activities : Provisioning , Production , Sales , Commercial Distribution.It also allows the correlation of Structured Data with Unstructured Data for knowledge generation.

Type Data :
Structured Data : DataCoSupplyChainDataset.csv
Unstructured Data : tokenized_access_logs.csv (Clickstream)

Types of Products : Clothing , Sports , and Electronic Supplies

Additionally it is attached in another file called DescriptionDataCoSupplyChain.csv, the description of each of the variables of the DataCoSupplyChainDatasetc.csv.

Acknowledgements & Source
Fabian Constante,

Fabian Constante
Instituto Politecnico de Leiria Escola Superior de Tecnologia e Gestao
Contribution: Master interested in topics related to Big Data
Fernando Silva, António Pereira

'''

df1= pd.read_csv('data/DATACOFinal_2R1.csv')
df2= pd.read_csv('data/DATACOFinal_2R2.csv')
df3= pd.read_csv('data/DATACOFinal_2R3.csv')
df_init=pd.concat([df1, df2,df3], 
                  axis = 1)


#df_init= pd.read_csv('data/DATACOFinal_2.csv')
df_fin = pd.read_csv('data/DATACOFinal_2.csv')
st.markdown('The initial raw data schema looks like this:')
st.table(df_init.head(5))
st.markdown('The data schema for the dashboard looks like this:')
st.table(df_fin.head(5))