#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
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


# for library_id in fov_include_GLOM:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata,
#         color="cellType",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_NZ:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata,
#         color="NeighborHoodsubType",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_NZ:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata,
#         color="cellType",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# # Plot imputed Expression

# In[14]:


adata_plot = ad.AnnData(adata_merge.layers["SCVI_imputed"])
adata_plot.layers["SCVI_Imputed"] = adata_plot.X.copy()

adata_plot.obs_names = adata_merge.obs_names
adata_plot.var_names = adata_merge.var_names

adata_plot.obs = adata.obs
adata_plot.uns = adata.uns
adata_plot.obsm = adata.obsm


# In[15]:


print(adata_plot)


# In[16]:


sc.pl.scatter(
    adata_plot,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="MEIS2",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

MEISvmax = 4


# In[17]:


sc.pl.scatter(
    adata_plot,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="LHX1",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)


# In[18]:


sc.pl.scatter(
    adata_plot,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="JAG1",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)


# In[19]:


sc.pl.scatter(
    adata_plot,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="EPO",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2,
)

APOEvmax = 20


# In[20]:


sc.pl.scatter(
    adata_plot,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="WT1",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

WT1vmax = 3


# In[21]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none' 


# for library_id in fov_include_GLOM:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="COL1A1",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 10,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# In[22]:


for library_id in fov_include_GLOM:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="COL3A1",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 0.25,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[23]:


for library_id in fov_include_GLOM:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="CFH",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 2,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# In[24]:


for library_id in fov_include_GLOM:
    print(library_id)
    ax = sq.pl.spatial_segment(
        adata_plot,
        color="ATP6V1A",
        library_key="fov",
        library_id=[library_id],
        seg_cell_id="cell_ID",
        seg_outline=True,
        img=False,
        title='',
        axis_label='',
        return_ax=True,
        frameon=False,
        vmax = 2,
        vmin = 0,
        colorbar = False,
        cmap = "plasma",
    )
    # Remove the legend, if present
    if ax.get_legend():
        ax.get_legend().remove()
    plt.show()


# for library_id in fov_include_GLOM:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="NPHS2",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 4,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_GLOM:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="PLVAP",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 2,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_NZ:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="TUBA1A",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 3,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
# 
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_NZ:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="COL1A1",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 10,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
# 
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# for library_id in fov_include_NZ:
#     print(library_id)
#     ax = sq.pl.spatial_segment(
#         adata_plot,
#         color="PLVAP",
#         library_key="fov",
#         library_id=[library_id],
#         seg_cell_id="cell_ID",
#         seg_outline=True,
#         img=False,
#         title='',
#         axis_label='',
#         return_ax=True,
#         frameon=False,
#         vmax = 2,
#         vmin = 0,
#         colorbar = False,
#         cmap = "plasma",
# 
#     )
#     # Remove the legend, if present
#     if ax.get_legend():
#         ax.get_legend().remove()
#     plt.show()

# In[25]:


ax = sq.pl.spatial_segment(
    adata_plot,
    color="WT1",
    library_key="fov",
    library_id=fov_include_NZ[-1],
    seg_cell_id="cell_ID",
    seg_outline=True,
    img=False,
    scalebar_dx=0.12,
    scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
    legend_fontsize='small',
    return_ax=True,
    cmap = "Greens",
    vmax = WT1vmax,
    vmin = 0,
    colorbar = False,
)
# Adjust the current figure to make space for the legend below
plt.gcf().subplots_adjust(bottom=1.2)  # You may need to adjust this value
# Get the handles and labels from the plot
handles, labels = ax.get_legend_handles_labels()
# Create the legend under the plot
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, frameon=False)  # Adjust ncol as needed
plt.show()


# In[26]:


ax = sq.pl.spatial_segment(
    adata_plot,
    color="ACE2",
    library_key="fov",
    library_id=fov_include_NZ[-1],
    seg_cell_id="cell_ID",
    seg_outline=True,
    img=False,
    scalebar_dx=0.12,
    scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
    legend_fontsize='small',
    return_ax=True,
    cmap = "Reds",
    vmax = 1,
    vmin = 0,
    colorbar = False,
)
# Adjust the current figure to make space for the legend below
plt.gcf().subplots_adjust(bottom=1.2)  # You may need to adjust this value
# Get the handles and labels from the plot
handles, labels = ax.get_legend_handles_labels()
# Create the legend under the plot
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, frameon=False)  # Adjust ncol as needed
plt.show()

