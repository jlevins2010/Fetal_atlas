#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module
import matplotlib.colors as mcolors

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
adata1.obs["cellType_SCANVI"] = adata.obs["cellType_SCANVI"]
print(adata1.obs["cellType_SCANVI"])

adataPT = adata1[adata1.obs["cellType_SCANVI"] == "PT"]
print(adataPT)


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
    gene_expression = adataPT.X[:, i].toarray().flatten()
    gene_expression = gene_expression[~np.isnan(gene_expression)]
    renew = adataPT.obs["renew"][~np.isnan(gene_expression)]
    dif = adataPT.obs["dif"][~np.isnan(gene_expression)]
    
    pt = adataPT.obs["pt"][~np.isnan(gene_expression)]
    podo = adataPT.obs["podocyte"][~np.isnan(gene_expression)]
    loh = adataPT.obs["loh"][~np.isnan(gene_expression)]
    dct = adataPT.obs["dct"][~np.isnan(gene_expression)]

    tubule = adataPT.obs["tubule"][~np.isnan(gene_expression)]
    glom = adataPT.obs["glom"][~np.isnan(gene_expression)]
    
    pseudo = adataPT.obs["pseudo"][~np.isnan(gene_expression)]

    
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


plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True


# In[11]:


colors = [
    (1, 0.5, 0),  # Orange
    (0.8, 0.8, 0.8),  # Grey
    (0, 0.5, 1),  # Blue
]

# Create the colormap
cmap = mcolors.LinearSegmentedColormap.from_list("orange_grey_blue", colors)

n, bins, patches = plt.hist(df['Glom'], bins=50, density = True)

# Normalize the bin values for color mapping
norm = plt.Normalize(bins.min(), bins.max())

# Color each bar based on its x-position
for patch, bin_value in zip(patches, bins):
    color = cmap(norm(bin_value))
    patch.set_facecolor(color)
    
agt_index = df[df['Ligand'] == "AGT"].index[0]
jag_index = df[df['Ligand'] == "JAG1"].index[0]
notch1_index = df[df['Ligand'] == "NOTCH1"].index[0]
notch2_index = df[df['Ligand'] == "NOTCH2"].index[0]


plt.annotate("AGT", xy=(df['Glom'][agt_index], 0.05), xytext=( 0, 50), textcoords="offset points", arrowprops=dict(arrowstyle="-"), fontsize=24)
plt.annotate("JAG1", xy=(df['Glom'][jag_index], 0.05), xytext=(20, 70), textcoords="offset points", arrowprops=dict(arrowstyle="-"), fontsize=24)
plt.annotate("NOTCH1", xy=(df['Glom'][notch1_index], 0.05), xytext=(-60, 30), textcoords="offset points", arrowprops=dict(arrowstyle="-"), fontsize=24)


plt.title("Glomerular Fate Probability")
plt.xlabel("Correlation")
plt.ylabel("Density")

# Show the plot
plt.show()


# In[12]:


df.to_csv('/home/levinsj/spatial/adata/project_Files/Fetal/ligandCorr_PT_only.csv')

