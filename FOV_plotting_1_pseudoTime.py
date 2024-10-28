#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import normalize

import scipy


# In[2]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/individual_Samples/FK1_raw.h5ad")


# In[3]:


adata.layers["counts"] = adata.X.copy()


# In[4]:


sc.pl.scatter(
    adata,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="fov",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

fov_include_GLOM = ['49', '50',  '51', '60', '61', '62', '71' , '72','73' ]
fov_include_NZ = ['79', '68',  '57', '46', '80', '69', '58' , '47' ]


# In[5]:


adata_merge = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/FK1_neighborhoodsCalled.h5ad")
adata_merge = adata_merge[adata_merge.obs["sample"] == "1"]
print(adata_merge)


# In[6]:


#adata_merge.obs = adata_merge.obs.set_index(adata_merge.obs.index.map(lambda x: f"c_{x}"))
adata_merge.obs.index = adata_merge.obs.index.str.replace('-1', '')
adata = adata[adata_merge.obs_names, :]


# In[7]:


print(adata.obs.index[0:5])


# In[8]:


print(adata_merge.obs.index[0:5])


# # Plot measured Expression

# In[9]:


adata.obs["cellType"] = adata_merge.obs["cellType_SCANVI"]
adata.obs["NeighborHoodsubType"] = adata_merge.obs["NeighborHoodsubType"]


# In[10]:


cell_types = {"DCT": "DCT",
               "Endothelium": "Endothelium",
               "UB_CT": "UB_CT",
               "Podocyte": "Podocyte", 
               "Stroma": "Stroma",
               "PT": "PT", 
               "Int": "Int",
               "Ureth": "Ureth", 
               "PEC": "PEC", 
               "LOH": "LOH",
               "Immune Cells": 'Immune Cells',
         }

neighborhoods = {"Blastema":"Blastema", 
               "PES":"PES",
               "EarlyGlom":"EarlyGlom",
               "Podocyte": "Podocyte", 
               "PT": "PT", 
               "PEC": "PEC", 
               "LOH": "LOH",
               "DCT": "DCT",
               "NephrogenicZoneOther":"NephrogenicZoneOther"}

adata.obs["cellType"] = adata.obs["cellType"].map(cell_types).astype('category')
adata.obs["NeighborHoodsubType"] = adata.obs["NeighborHoodsubType"].map(neighborhoods).astype('str')
adata.obs["NeighborHoodsubType"] = adata.obs["NeighborHoodsubType"].fillna("Other").astype('category')
print(adata.obs["NeighborHoodsubType"].value_counts())


# In[11]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none' 


# In[12]:


neighbor_colors = {
    "DCT": (255/255, 127/255, 14/255),
    "PES": (227/255, 119/255, 194/255),
    "Blastema": (31/255, 119/255, 180/255),  
    "Podocyte": (188/255, 189/255, 34/255),
    "EarlyGlom":(44/255, 160/255, 44/255),
    "PT": (127/255, 127/255, 127/255),
    "PEC": (140/255, 86/255, 75/255),
    "LOH": (214/255, 39/255, 40/255),
    "nan": (220/255, 220/255, 220/255),
    "NephrogenicZoneOther": (148/255, 103/255, 189/255)
}

# Assuming 'adata_subset' is your AnnData object with the annotation
cell_type_order = adata.obs["NeighborHoodsubType"].cat.categories.tolist()
colors_mapped = [neighbor_colors[cell_type] for cell_type in cell_type_order]
adata.uns['NeighborHoodsubType_colors'] = colors_mapped


# In[13]:


cell_colors = {
    "DCT": (128/255, 5/255, 21/255),
    "Endothelium": (122/255, 224/255, 49/255),
    "UB_CT": (1, 1, 1),  
    "Podocyte": (173/255, 156/255, 0),
    "Stroma": (121/255, 75/255, 130/255),
    "NPC": (255/255, 128/255, 0),
    "PT": (255/255, 0, 212/255),
    "Int": (105/255, 140/255, 255/255),
    "Ureth": (212/255, 114/255, 34/255),
    "PEC": (255/255, 0, 17/255),
    "LOH": (35/255, 94/255, 0),
    "Immune Cells": (117/255, 117/255, 117/255),
}

# Assuming 'adata_subset' is your AnnData object with the annotation
cell_type_order = adata.obs["cellType"].cat.categories.tolist()
colors_mapped = [cell_colors[cell_type] for cell_type in cell_type_order]
adata.uns['cellType_colors'] = colors_mapped


# In[14]:


adata = adata[adata_merge.obs_names, :]

var_to_plot = ["PseudoTime_SCVI", "DCT_absorbtion_SCVI", "LOH_absorbtion_SCVI", "PT_absorbtion_SCVI", "Podo_absorbtion_SCVI", "NPC_SCVI", "Tubule_SCVI","Glomerular_SCVI","Differentiated Cell_SCVI"]
print(adata_merge.obs[var_to_plot])

absDataframe = adata_merge.obs[var_to_plot]

adata_plot = ad.AnnData(scipy.sparse.csr_matrix(absDataframe))
adata_plot.obs_names = adata_merge.obs_names
adata_plot.var_names = var_to_plot

adata_plot.uns = adata.uns
adata_plot.obsm = adata.obsm
adata_plot.obsp = adata.obsp
adata_plot.obs = adata.obs


# In[15]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none' 


# In[16]:


for library_id in fov_include_NZ:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="PseudoTime_SCVI",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 0.6,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[17]:


for library_id in fov_include_NZ:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="PT_absorbtion_SCVI",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 0.8,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[18]:


for library_id in fov_include_NZ:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="DCT_absorbtion_SCVI",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 1,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[19]:


for library_id in fov_include_NZ:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="LOH_absorbtion_SCVI",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 1,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[20]:


for library_id in fov_include_NZ:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="Podo_absorbtion_SCVI",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 1,
        vmin = 0.3,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()

