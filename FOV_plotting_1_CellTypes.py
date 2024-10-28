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


# In[9]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'pad_inches':0}")

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['savefig.facecolor'] = 'black' 


# ## Podocyte Plot

# In[10]:


adata.obs["cellType"] = adata_merge.obs["cellType_SCANVI"]


# cell_types = {"DCT": "DCT",
#                "Endothelium": "Endothelium",
#                "UB_CT": "UB_CT",
#                "Podocyte": "Podocyte", 
#                "Stroma": "Stroma",
#                "PT": "PT", 
#                "Int": "Int",
#                "Ureth": "Ureth", 
#                "PEC": "PEC", 
#                "LOH": "LOH",
#                "Immune Cells": 'Immune Cells',
#          }

# In[11]:


cell_types = {"DCT": "Other",
               "Endothelium": "Other",
               "UB_CT": "UB_CT",
               "Podocyte": "Other", 
               "Stroma": "Stroma",
               "PT": "Other", 
               "Int": "Other",
               "Ureth": "Other", 
               "PEC": "Other", 
               "LOH": "Other",
               "Immune Cells": 'Other',
               "Ureth": "Ureth",
         }

adata.obs["cellType"] = adata.obs["cellType"].map(cell_types).astype('category')


# In[12]:


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
    "Other": (40/255, 40/255, 40/255)
}

# Assuming 'adata_subset' is your AnnData object with the annotation
cell_type_order = adata.obs["cellType"].cat.categories.tolist()
colors_mapped = [cell_colors[cell_type] for cell_type in cell_type_order]
adata.uns['cellType_colors'] = colors_mapped


# In[13]:


for library_id in fov_include_GLOM:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata,
        color="cellType",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout(pad=0)
    plt.show()
    

