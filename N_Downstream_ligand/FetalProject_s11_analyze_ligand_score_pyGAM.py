#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module

from pygam import GAM, s, te

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


print(adata1)
print(adata1.obs["renew"])

adata1 = adata1[~adata1.obs["renew"].isna()]


# In[9]:


ligand_score = adata1.X[:, adata1.var.index == "VEGFA"].todense()
print(ligand_score)
print(len(ligand_score))


# In[10]:


def Plot_4_lineages_Ligands(goi, adata_input):
    model = GAM(s(0,spline_order=3,n_splines = 6), distribution='normal', max_iter=2000, fit_intercept=True,  tol=0.001, verbose=False)
    
    x_param = 'pseudo' 
    lineage_1 = 'loh'
    lineage_2 = 'pt'
    lineage_3 = 'dct'
    lineage_4 = 'podocyte'
    quantile_max = 0.95

    ###
    latent_max_1 = adata_input[adata_input.obs[lineage_1] >= 0.9].obs[x_param].quantile(quantile_max)
    adata_plot_1 = adata_input[adata_input.obs[x_param] < latent_max_1]
    
    X1 = adata_plot_1.obs[[x_param]].to_numpy()
    y1 = adata_plot_1.X[:, adata_plot_1.var.index == goi].todense()
    w1 = adata_plot_1.obs[lineage_1].to_numpy()

    model1 = model
    model1.fit(X1,y1,w1)
    XX1 = np.linspace(0, latent_max_1, 500)
    y_pred1 = model1.predict(XX1)

    ###
    latent_max_2 = adata_input[adata_input.obs[lineage_2] >= 0.9].obs[x_param].quantile(quantile_max)
    adata_plot_2 = adata_input[adata_input.obs[x_param] < latent_max_2]

    X2 = adata_plot_2.obs[[x_param]].to_numpy()
    y2 = adata_plot_2.X[:, adata_plot_2.var.index == goi].todense()
    w2 = adata_plot_2.obs[lineage_2].to_numpy()

    model2 = model
    model2.fit(X2,y2,w2)
    XX2 = np.linspace(0, latent_max_2, 500)
    y_pred2 = model2.predict(XX2)

    ###
    latent_max_3 = adata_input[adata_input.obs[lineage_3] >= 0.9].obs[x_param].quantile(quantile_max)
    adata_plot_3 = adata_input[adata_input.obs[x_param] < latent_max_3]
    
    X3 = adata_plot_3.obs[[x_param]].to_numpy()
    y3 = adata_plot_3.X[:, adata_plot_3.var.index == goi].todense()
    w3 = adata_plot_3.obs[lineage_3].to_numpy()

    model3 = model
    model3.fit(X3,y3,w3)
    XX3 = np.linspace(0, latent_max_3, 500)
    y_pred3 = model3.predict(XX3)

    ###
    latent_max_4 = adata_input[adata_input.obs[lineage_4] >= 0.9].obs[x_param].quantile(quantile_max)
    adata_plot_4 = adata_input[adata_input.obs[x_param] < latent_max_4]

    X4 = adata_plot_4.obs[[x_param]].to_numpy()
    y4 = adata_plot_4.X[:, adata_plot_4.var.index == goi].todense()
    w4 = adata_plot_4.obs[lineage_4].to_numpy()

    model4 = model
    model4.fit(X4,y4,w4)
    XX4 = np.linspace(0, latent_max_4, 500)
    y_pred4 = model4.predict(XX4)
    ###

    plt.plot(XX1, y_pred1, color='red', label='LOH')
    plt.plot(XX2, y_pred2, color='blue', label='PT')
    plt.plot(XX3, y_pred3, color='black', label='DCT')
    plt.plot(XX4, y_pred4, color='darkorange', label='Podocyte')
    
    plt.xlim(0, 0.8)
    plt.show()


# In[11]:


Plot_4_lineages_Ligands("VEGFA", adata1)
Plot_4_lineages_Ligands("IGF2", adata1)
Plot_4_lineages_Ligands("AVP", adata1)
Plot_4_lineages_Ligands("WNT4", adata1)





# In[12]:


Plot_4_lineages_Ligands("FGF9", adata1)
Plot_4_lineages_Ligands("KNG1", adata1)
Plot_4_lineages_Ligands("ACE2", adata1)
Plot_4_lineages_Ligands("EFNA4", adata1)


# In[13]:


Plot_4_lineages_Ligands("ACE2", adata1)
Plot_4_lineages_Ligands("ITGA2", adata1)
Plot_4_lineages_Ligands("AGT", adata1)
Plot_4_lineages_Ligands("LGR5", adata1)


# In[14]:


Plot_4_lineages_Ligands("ITGA8", adata1)
Plot_4_lineages_Ligands("RSPO3", adata1)
Plot_4_lineages_Ligands("GDNF", adata1)
Plot_4_lineages_Ligands("WNT7B", adata1)


# In[15]:


Plot_4_lineages_Ligands("VEGFB", adata1)


# In[16]:


Plot_4_lineages_Ligands("KL", adata1)

