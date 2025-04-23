#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import pandas as pd
import anndata as ad

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy


# Set parameters

# In[2]:


neighbors = 15

colors = {"DCT": "#800515",
               "Endothelium": "#7ae031",
               "UB_CT": "black",
               "Podocyte": "#ad9c00", 
               "Stroma": "#794b82",
               "NPC": "#ff8000", 
               "PT": "#ff00d4", 
               "Int": "#698cff",
               "Ureth": "#d47222", 
               "PEC": "#ff0011", 
               "LOH": "#235e00",
               "Immune Cells": '#757575',
         }


# # FK1

# In[3]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
adata = adata[adata.obs["sample"] == "1"]


# In[4]:


print(adata.obs["sample"].value_counts())


# ### Need to export the counts as a sparse matrix

# In[5]:


from scipy import sparse, io


# In[6]:


m = adata.layers["SCVI_imputed"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/FK1_imputed.mtx", m)
adata.obs.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK1_obs_names.csv', index=True)

m = adata.layers["counts"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/FK1_rawCounts.mtx", m)


# In[7]:


probeNames = pd.read_csv('/home/levinsj/spatial/adata/probeNames.csv')

adata.var["gene_Name"] = adata.var.index
adata.var["CosMx"] = adata.var["gene_Name"].isin(probeNames["x"].tolist())
print(adata.var.CosMx.value_counts())


# In[8]:


adata.var.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK1_var_names.csv', index=True)


# # HK3524

# In[9]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
adata = adata[adata.obs["sample"] == "0"]


# In[10]:


print(adata.obs["sample"].value_counts())


# In[11]:


m = adata.layers["SCVI_imputed"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/HK3524_imputed.mtx", m)
adata.obs.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/HK3524_obs_names.csv', index=True)

m = adata.layers["counts"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/HK3524_rawCounts.mtx", m)


# In[12]:


probeNames = pd.read_csv('/home/levinsj/spatial/adata/probeNames.csv')

adata.var["gene_Name"] = adata.var.index
adata.var["CosMx"] = adata.var["gene_Name"].isin(probeNames["x"].tolist())
print(adata.var.CosMx.value_counts())


# In[13]:


adata.var.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/HK3524_var_names.csv', index=True)


# # FK4

# In[14]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
adata = adata[adata.obs["sample"] == "2"]


# In[15]:


print(adata.obs["sample"].value_counts())


# In[16]:


m = adata.layers["SCVI_imputed"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/FK4_imputed.mtx", m)
adata.obs.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK4_obs_names.csv', index=True)

m = adata.layers["counts"]
io.mmwrite("/home/levinsj/spatial/adata/Export_to_R_files/FK4_rawCounts.mtx", m)


# In[17]:


probeNames = pd.read_csv('/home/levinsj/spatial/adata/probeNames.csv')

adata.var["gene_Name"] = adata.var.index
adata.var["CosMx"] = adata.var["gene_Name"].isin(probeNames["x"].tolist())
print(adata.var.CosMx.value_counts())


# In[18]:


adata.var.to_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK4_var_names.csv', index=True)

