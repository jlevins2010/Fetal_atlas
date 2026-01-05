#!/usr/bin/env python
# coding: utf-8

# ## Further cleaning of the integrated object
# 

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import matplotlib as mpl
import os
import os.path
import torch
import scvi
from IPython.display import display


# In[2]:


colors = {"DCT": "#8b171a",
               "Endothelium": "#68bd45",
               "Podocyte": "#ac9f34", 
               "Stroma": "#834786",
               "PT": "#da3d96", 
               "Int": "#6b81c0",
               "PEC": "#ed1c24", 
               "LOH": "#1b6131",
               "Immune Cells": '#767576',
              "UE":"#000000" 
         }


# ### scanpy variables

# In[3]:


sc.settings.verbosity = 3
sc.logging.print_header()
mpl.rcParams['figure.dpi'] = 450
min_distUMAP = 0.2
res = 2.0
n_features = 250000


# ### SCVI import

# In[4]:


batch_size = 32
max_epochs = 2000
earlyStopping = True
learning_rate = 0.0005
scvi_layers = 3
scvi_latent = 30
fraction_expressed = 0.01

SCVI_LATENT_KEY = "X_scvi"
SCVI_CLUSTERS_KEY = "clusters_scvi"
SCANVI_LATENT_KEY = "X_scANVI"


# ### SCANPY Settings

# In[5]:


sc.settings.verbosity = 3
sc.logging.print_header()
mpl.rcParams['figure.dpi'] = 450
min_distUMAP = 0.2


# In[6]:


def keep_matching_elements(my_list, adata_var_names):
  return [element for element in my_list if element in adata_var_names]


# In[7]:


input_file="/vast/projects/chenyuli/nephrobase/levinsj/adata/fetal_annotated_AllCells_withXenium_postSCANVI_pre_anno.h5ad"
output_File="/vast/projects/chenyuli/nephrobase/levinsj/adata/fetal_annotated_AllCells_withXenium_postSCANVI_post_anno.h5ad"


# In[8]:


adata = sc.read_h5ad(input_file)
print(adata)
print(adata.obs.index)


# In[9]:


sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=min_distUMAP)
sc.tl.leiden(adata, key_added=SCVI_CLUSTERS_KEY, resolution=res)
sc.pl.umap(adata, color = SCVI_CLUSTERS_KEY, legend_loc='on data', frameon = False, legend_fontsize=10, legend_fontoutline=2)


# In[10]:


adata = adata[adata.obs[SCVI_CLUSTERS_KEY] != "40"]
adata.obs["cellType"] = adata.obs[SCVI_CLUSTERS_KEY]

cell_identities = {'0':'PT','1':'Stroma','2':'Endothelium','3':'Int','4':'Stroma','5':'Stroma','6':'LOH',
                   '7':'Podocyte','8':'DCT','9':'Stroma','10':'LOH','11':'Podocyte','12':'UE','13':'Stroma','14':'Endothelium',
                   '15':'LOH', '16':'Int','17':'UE','18':'Stroma','19':'Int','20':'LOH','21':'PT', '22':'PT',
                   '23':'Podocyte','24':'PT','25':'Stroma','26':'Int','27':'Stroma', '28':'Int','29':'Endothelium', '30':'LOH',
                   '31':'Int','32':'Int','33': 'DCT', '34':'UE','35':'PT', '36':'LOH','37':'Stroma',
                   '38':'UE','39':'Endothelium','41':'Immune Cells','42':'PT','43':'Stroma','44':'PEC',
                   '45':'LOH','46':'Stroma','47':'LOH','48':'Stroma','49':'UE', '50':'Int','51':'Stroma','52':'Int', '53':'DCT','54':'UE', 
                   '55':'DCT','56':'Stroma','57':'Stroma','58':'Immune Cells','59':'UE','60':'Podocyte',"61":'PEC','62':'Int',
                   '63':'DCT','64':'Int','65':'PT','66':'Endothelium','67':'Endothelium','68':'DCT','69':'Int','70':'Immune Cells'}


adata.obs["cellType"] = adata.obs[SCVI_CLUSTERS_KEY].map(cell_identities).astype('category')


# In[11]:


sc.pl.umap(
    adata,
    color="cellType",
    legend_loc='on data',
    frameon=False,
    legend_fontsize=10,
    legend_fontoutline=2,
    palette = colors)

sc.pl.umap(
    adata,
    color="type",
    legend_fontsize=10,
    legend_fontoutline=2)

sc.pl.umap(
    adata,
    color="tech",
    frameon=False,
    legend_fontsize=10,
    legend_fontoutline=2)


# In[12]:


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata) 
sc.tl.rank_genes_groups(adata, "cellType", method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)


# In[13]:


var_names = ['JAG1', 'STMN1', 'NPHS2', 'WT1',"VIM","VCAM1","APOE","LRP2",'UMOD', 'SLC12A1',
             'SLC8A1', 'SLC12A3', 'AQP2', 'GATA3', 'PLVAP', 'PECAM1', 'COL1A1', 'COL3A1', 'HLA-DRA','HLA-DPA1']
order = ["Int","Podocyte","PEC", "PT","LOH","DCT","UE","Endothelium","Stroma","Immune Cells"]


# In[14]:


sc.pl.dotplot(
    adata,
    var_names=keep_matching_elements(var_names, adata.var_names),
    groupby='cellType',
    categories_order=order,
    standard_scale='var',
    cmap='Blues',
    dot_max=0.6,
    dot_min=0.0,
    figsize=(8, 6),
    dendrogram=False,
    log=True,
)



# ### save files

# In[15]:


adata.write(output_File)
print(adata.obs["sample"].value_counts())


# In[16]:


output_csv = "/vast/projects/chenyuli/nephrobase/levinsj/adata/HK3888_celltype.csv"

filter_mask = adata.obs['sample'] == "HK3888"

filtered_obs = adata.obs[filter_mask]

export_df = pd.DataFrame({
    'Barcode': filtered_obs.index,  # Use the index of .obs for Barcodes
    'CellType': filtered_obs['cellType']
})

export_df.to_csv(output_csv, index=False)


# In[17]:


output_csv = "/vast/projects/chenyuli/nephrobase/levinsj/adata/22020263005_celltype.csv"

filter_mask = adata.obs['sample'] == "22020263005"

filtered_obs = adata.obs[filter_mask]

export_df = pd.DataFrame({
    'Barcode': filtered_obs.index,  # Use the index of .obs for Barcodes
    'CellType': filtered_obs['cellType']
})

export_df.to_csv(output_csv, index=False)


# In[ ]:


output_csv = "/vast/projects/chenyuli/nephrobase/levinsj/adata/CosMx_celltype.csv"

filter_mask = adata.obs['tech'] == "CosMx"

filtered_obs = adata.obs[filter_mask]

export_df = pd.DataFrame({
    'Barcode': filtered_obs.index,  # Use the index of .obs for Barcodes
    'CellType': filtered_obs['cellType'],
    'Sample': filtered_obs["sample"]
})

export_df.to_csv(output_csv, index=False)

