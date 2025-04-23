#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# In[2]:


mpl.rcParams['figure.dpi'] = 450


# Set parameters

# In[3]:


neighbors = 15
fineNeighborhood = 4
coarseNeighborhood = 3

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
               "Nephron":"#698cff",
         }


# In[4]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)

adata_UB = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_Abs_UB_Only.h5ad")
adata_UB = adata_UB[adata_UB.obs["UB_absorption_SCVI"] > 0.5]

common_obs_idx = adata.obs_names.intersection(adata_UB.obs_names)
adata = adata[~adata.obs_names.isin(common_obs_idx)]

print(adata)


# In[5]:


print(adata.obs_names)


# In[6]:


adata1 = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/ligandScores_all_ligands.h5ad")


# In[7]:


print(adata1.obs_names)


# In[8]:


adata1.obs["renew"] = adata.obs["NPC_SCVI"]
adata1.obs["dif"] = adata.obs['Differentiated Cell_SCVI']

adata1.obs["podocyte"] = adata.obs["Podo_absorbtion_SCVI"]
adata1.obs["pt"] = adata.obs["PT_absorbtion_SCVI"]
adata1.obs["loh"] = adata.obs["LOH_absorbtion_SCVI"]
adata1.obs["dct"] = adata.obs["DCT_absorbtion_SCVI"]

adata1.obs["tubule"] = adata.obs["Tubule_SCVI"]
adata1.obs["glom"] = adata.obs["Glomerular_SCVI"]

adata1.obs["pseudo"] = adata.obs["PseudoTime_SCVI"]
adata1.obs["cellType"] = adata.obs["cellType"]


print(adata1)
print(adata1.obs["renew"])

adata1 = adata1[~adata1.obs["renew"].isna()]


# In[9]:


renew_correlations = []
dif_correlations = []

podo_correlations = []
pt_correlations = []
loh_correlations = []
dct_correlations = []

tubule_correlations = []
glom_correlations = []

pseudo_correlation = []

for i in range(adata1.shape[1]):
    gene_expression = adata1.X[:, i].toarray().flatten()
    gene_expression = gene_expression[~np.isnan(gene_expression)]
    renew = adata1.obs["renew"][~np.isnan(gene_expression)]
    dif = adata1.obs["dif"][~np.isnan(gene_expression)]
    
    pt = adata1.obs["pt"][~np.isnan(gene_expression)]
    podo = adata1.obs["podocyte"][~np.isnan(gene_expression)]
    loh = adata1.obs["loh"][~np.isnan(gene_expression)]
    dct = adata1.obs["dct"][~np.isnan(gene_expression)]

    tubule = adata1.obs["tubule"][~np.isnan(gene_expression)]
    glom = adata1.obs["glom"][~np.isnan(gene_expression)]
    
    pseudo = adata1.obs["pseudo"][~np.isnan(gene_expression)]

    
    # Calculate correlations
    correlation_renew = np.corrcoef(gene_expression, renew)[0, 1]
    correlation_dif = np.corrcoef(gene_expression, dif)[0, 1]
    
    correlation_pt = np.corrcoef(gene_expression, pt)[0, 1]
    correlation_podo = np.corrcoef(gene_expression, podo)[0, 1]
    correlation_loh = np.corrcoef(gene_expression, loh)[0, 1]
    correlation_dct = np.corrcoef(gene_expression, dct)[0, 1]

    correlation_glom = np.corrcoef(gene_expression, glom)[0, 1]
    correlation_tub = np.corrcoef(gene_expression, tubule)[0, 1]
    
    correlation_pseudo = np.corrcoef(gene_expression, pseudo)[0, 1]
    
    renew_correlations.append(correlation_renew)
    dif_correlations.append(correlation_dif)
    
    pt_correlations.append(correlation_pt)
    podo_correlations.append(correlation_podo)
    loh_correlations.append(correlation_loh)
    dct_correlations.append(correlation_dct)
    
    glom_correlations.append(correlation_glom)
    tubule_correlations.append(correlation_tub)

    pseudo_correlation.append(correlation_pseudo)

df = pd.DataFrame({'Ligand': adata1.var_names.tolist(), 'Renew': renew_correlations, "Diff": dif_correlations,
                  "PT": pt_correlations, "Podocyte": podo_correlations, "LOH": loh_correlations, "DCT": dct_correlations,
                  "Tubule": tubule_correlations, "Glom": glom_correlations, "PseudoTime": pseudo_correlation})

print(df)


# In[10]:


df.to_csv('/home/levinsj/spatial/adata/project_Files/Fetal/ligandCorr.csv')


# In[11]:


plt.hist(df['Renew'], bins=30, density=True)

# Annotate specific observations (replace with your actual indices)

#igf2_index = df.iloc[df['Ligand'] == "P01344"].index[0]
#gdnf_index = df.iloc[df['Ligand'] == "P56159"].index[0]#

#plt.annotate("IGF2", xy=(df['Renew'][igf2_index], 0.05), xytext=(5, 5), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
#lt.annotate("GDNF", xy=(df['Renew'][gdnf_index], 0.05), xytext=(-5, 5), textcoords="offset points", arrowprops=dict(arrowstyle="->"))

# Customize the plot (optional)
plt.title("Renewal")
plt.xlabel("Correlation")
plt.ylabel("Density")

# Show the plot
plt.show()

