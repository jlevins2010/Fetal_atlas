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


df = pd.read_csv('/home/levinsj/spatial/adata/project_Files/FK1_cells_manual_neighborhoods.csv', index_col = "Name")

print(df["type"].value_counts())


# In[5]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)


# In[6]:


adata1 = adata[adata.obs["sample"] == "1"]

sc.pl.scatter(
    adata1,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="cellType_SCANVI",
    size=1,
)


# In[7]:


NephrogenicZone = df[df["type"] == "NephrogenicZone"]
Blastema = df[df["type"] == "Blastema"]
PES = df[df["type"] == "PES"]
EarlyGlom = df[df["type"] == "EarlyGlom"]
EarlyGlom = EarlyGlom.loc[~EarlyGlom.index.duplicated()]
EarlyGlom.index = EarlyGlom["Object ID"]


# In[8]:


adata1.obs["NephrogenicZone"] = NephrogenicZone["type"]
adata1.obs["Blastema"] = Blastema["type"]
adata1.obs["PES"] = PES["type"]
adata1.obs["EarlyGlom"] = EarlyGlom["type"]


# In[9]:


adata1.obs["Podocyte"] = np.where(adata1.obs["cellType_SCANVI"] == "Podocyte", "Podocyte", pd.NA)
adata1.obs["PT"] = np.where(adata1.obs["cellType_SCANVI"] == "PT", "PT", pd.NA)
adata1.obs["PEC"] = np.where(adata1.obs["cellType_SCANVI"] == "PEC", "PEC", pd.NA)
adata1.obs["LOH"] = np.where(adata1.obs["cellType_SCANVI"] == "LOH", "LOH", pd.NA)
adata1.obs["DCT"] = np.where(adata1.obs["cellType_SCANVI"] == "DCT", "DCT", pd.NA)
adata1.obs["Nephrogenic_Zone_other"] = np.where(adata1.obs["NephrogenicZone"] == "NephrogenicZone", "NephrogenicZoneOther", pd.NA)



print(adata1.obs["Podocyte"].value_counts())


# In[10]:


adata1.obs["NephrogenicZone"] = adata1.obs["NephrogenicZone"].fillna("Other")
adata1.obs["NeighborHoodsubType"] = adata1.obs["Blastema"].fillna(adata1.obs["PES"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["EarlyGlom"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["Nephrogenic_Zone_other"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["Podocyte"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["PT"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["PEC"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["LOH"])
adata1.obs["NeighborHoodsubType"] = adata1.obs["NeighborHoodsubType"].fillna(adata1.obs["DCT"])


print(adata1.obs["NephrogenicZone"].value_counts())
print(adata1.obs["NeighborHoodsubType"].value_counts())


# In[11]:


sc.pl.scatter(
    adata1,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="NephrogenicZone",
    size=1,
)

sc.pl.scatter(
    adata1,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="NeighborHoodsubType",
    size=4,
)


# In[12]:


sq.gr.spatial_neighbors(
    adata1,
    radius=25/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "25_spatial"
)

sq.gr.spatial_neighbors(
    adata1,
    radius=50/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "50_spatial"
)

sq.gr.spatial_neighbors(
    adata1,
    radius=75/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "75_spatial"
)

sq.gr.spatial_neighbors(
    adata1,
    radius=100/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "100_spatial"
)


# In[13]:


sc.pl.violin(adata1, keys='PseudoTime_SCVI', groupby='NephrogenicZone', rotation=90, size = 0)
sc.pl.violin(adata1, keys='NPC_SCVI', groupby='NephrogenicZone', rotation=90, size = 0)
sc.pl.violin(adata1, keys='G2M_score', groupby='NephrogenicZone', rotation=90, size = 0)
sc.pl.violin(adata1, keys='S_score', groupby='NephrogenicZone', rotation=90, size = 0)


# In[14]:


groupby_order = ['Blastema','NephrogenicZoneOther', 'PES', 'EarlyGlom',"PEC","Podocyte","PT","LOH","DCT"]


# In[15]:


sc.pl.violin(adata1, keys='PseudoTime_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='NPC_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='Differentiated Cell_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)

sc.pl.violin(adata1, keys='Tubule_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='Glomerular_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)

sc.pl.violin(adata1, keys='Podo_absorbtion_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='PT_absorbtion_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='LOH_absorbtion_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='DCT_absorbtion_SCVI', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)

sc.pl.violin(adata1, keys='G2M_score', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)
sc.pl.violin(adata1, keys='S_score', groupby='NeighborHoodsubType', rotation=90, size = 0, order = groupby_order)


# In[16]:


dist = pd.read_csv('/home/levinsj/spatial/adata/project_Files/FK1_cells_distance_to_border.csv', index_col = "Name")

dist.head(5)


# In[17]:


adata1.obs["dist_to_border"] = dist["distance_to_border"].astype(int)

sc.pl.scatter(
    adata1,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="dist_to_border",
    size=1,
)


# In[18]:


dist = pd.read_csv('/home/levinsj/spatial/adata/project_Files/FK1_cells_distance_to_Nephrogenic_Zone.csv', index_col = "Name")

dist.head(5)


# In[19]:


adata1.obs["dist_to_NephrogenicZone"] = dist["min_distance_to_polygons"].astype(int)

sc.pl.scatter(
    adata1,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="dist_to_NephrogenicZone",
    size=1,
)


# In[20]:


adata1.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/FK1_neighborhoodsCalled.h5ad")

print(adata1)


# In[21]:


cell_types = {"DCT": "Nephron",
               "Endothelium": "Endothelium",
               "UB_CT": "UB_CT",
               "Podocyte": "Nephron", 
               "Stroma": "Stroma",
               "PT": "Nephron", 
               "Int": "Nephron",
               "PEC": "Nephron", 
               "LOH": "Nephron"
         }

adata1.obs["cellType_SCANVI_2"] = adata1.obs["cellType_SCANVI"].map(cell_types).astype('category')

adata1_df = adata1[~adata1.obs["cellType_SCANVI"].isin(["Ureth","Immune Cells"])]



# In[22]:


df = adata1_df.obs["NeighborHoodsubType"].value_counts().to_frame()
df.index = adata1_df.obs["NeighborHoodsubType"].value_counts().index

df['sampleBreakdown'] = object
df['neighbors'] = object
df['composition'] = object

df = df.sort_index(ascending=True)

lst = ["Blastema", "NephrogenicZoneOther", "PES","EarlyGlom","PEC","Podocyte","PT","LOH","DCT"]
df = df.loc[lst]


# In[23]:


## Neighborhoods 25 micron


# In[24]:


for i in df.index:   
    type_array = (np.where(adata1_df.obs["NeighborHoodsubType"]==i)[0])
    ## find nearest neighbors
    connectivities_cellType = adata1_df.obsp["25_spatial_connectivities"][type_array,:] # make an matrix with just cell type i comprising rows
    sum = connectivities_cellType.sum()
    count = df['count'].loc[i]
    nn = []
    for j_index, j in enumerate(adata1_df.obs["cellType_SCANVI_2"].unique()):
        type2_array = (np.where(adata1_df.obs["cellType_SCANVI_2"]==j)[0])
        nn.append(connectivities_cellType[:,type2_array].sum()/(count))
        #nn.append(connectivities_cellType[:,type2_array].sum()/(sum))
        df.at[i,"neighbors"] = nn


# In[25]:


idx = [0, 2, 1, 3]
mylabels = "Nephron","UB_CT","Stroma","Endothelium"


# In[26]:


for i in df.index:   
    type_array = (np.where(adata1_df.obs["NeighborHoodsubType"]==i)[0])
    ## find nearest neighbors
    connectivities_cellType = adata1_df.obsp["25_spatial_connectivities"][type_array,:] # make an matrix with just cell type i comprising rows
    sum = connectivities_cellType.sum()
    count = df['count'].loc[i]
    nn = []
    for j_index, j in enumerate(adata1_df.obs["cellType_SCANVI_2"].unique()):
        type2_array = (np.where(adata1_df.obs["cellType_SCANVI_2"]==j)[0])
        nn.append(connectivities_cellType[:,type2_array].sum()/(count))
        #nn.append(connectivities_cellType[:,type2_array].sum()/(sum))
        df.at[i,"neighbors"] = nn
        
# Blastema
idx = [0, 2, 1, 3]
mylabels = ["Nephron", "UB_CT", "Stroma", "Endothelium"]
SBlastema = df.neighbors[0]
SBlastema = [SBlastema[i] for i in idx]
y = np.array(SBlastema)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SOther = df.neighbors[1]
SOther = [SOther[i] for i in idx]
y = np.array(SOther)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPES = df.neighbors[2]
SPES = [SPES[i] for i in idx]
y = np.array(SPES)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


SEarlyGlom = df.neighbors[3]
SEarlyGlom = [SEarlyGlom[i] for i in idx]
y = np.array(SEarlyGlom)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPodo = df.neighbors[5]
SPodo = [SPodo[i] for i in idx]
y = np.array(SPodo)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPT = df.neighbors[6]
SPT = [SPT[i] for i in idx]
y = np.array(SPT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show() 

SLOH = df.neighbors[7]
SLOH = [SLOH[i] for i in idx]
y = np.array(SLOH)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SDCT = df.neighbors[8]
SDCT = [SDCT[i] for i in idx]
y = np.array(SDCT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


# In[27]:


## neighborhoods 50 micron


# In[28]:


for i in df.index:   
    type_array = (np.where(adata1_df.obs["NeighborHoodsubType"]==i)[0])
    ## find nearest neighbors
    connectivities_cellType = adata1_df.obsp["50_spatial_connectivities"][type_array,:] # make an matrix with just cell type i comprising rows
    sum = connectivities_cellType.sum()
    count = df['count'].loc[i]
    nn = []
    for j_index, j in enumerate(adata1_df.obs["cellType_SCANVI_2"].unique()):
        type2_array = (np.where(adata1_df.obs["cellType_SCANVI_2"]==j)[0])
        nn.append(connectivities_cellType[:,type2_array].sum()/(count))
        #nn.append(connectivities_cellType[:,type2_array].sum()/(sum))
        df.at[i,"neighbors"] = nn
        
# Blastema
idx = [0, 2, 1, 3]
mylabels = ["Nephron", "UB_CT", "Stroma", "Endothelium"]
SBlastema = df.neighbors[0]
SBlastema = [SBlastema[i] for i in idx]
y = np.array(SBlastema)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SOther = df.neighbors[1]
SOther = [SOther[i] for i in idx]
y = np.array(SOther)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPES = df.neighbors[2]
SPES = [SPES[i] for i in idx]
y = np.array(SPES)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


SEarlyGlom = df.neighbors[3]
SEarlyGlom = [SEarlyGlom[i] for i in idx]
y = np.array(SEarlyGlom)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPodo = df.neighbors[5]
SPodo = [SPodo[i] for i in idx]
y = np.array(SPodo)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPT = df.neighbors[6]
SPT = [SPT[i] for i in idx]
y = np.array(SPT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show() 

SLOH = df.neighbors[7]
SLOH = [SLOH[i] for i in idx]
y = np.array(SLOH)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SDCT = df.neighbors[8]
SDCT = [SDCT[i] for i in idx]
y = np.array(SDCT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


# In[29]:


## 75 micron

for i in df.index:   
    type_array = (np.where(adata1_df.obs["NeighborHoodsubType"]==i)[0])
    ## find nearest neighbors
    connectivities_cellType = adata1_df.obsp["75_spatial_connectivities"][type_array,:] # make an matrix with just cell type i comprising rows
    sum = connectivities_cellType.sum()
    count = df['count'].loc[i]
    nn = []
    for j_index, j in enumerate(adata1_df.obs["cellType_SCANVI_2"].unique()):
        type2_array = (np.where(adata1_df.obs["cellType_SCANVI_2"]==j)[0])
        nn.append(connectivities_cellType[:,type2_array].sum()/(count))
        #nn.append(connectivities_cellType[:,type2_array].sum()/(sum))
        df.at[i,"neighbors"] = nn
        
# Blastema
idx = [0, 2, 1, 3]
mylabels = ["Nephron", "UB_CT", "Stroma", "Endothelium"]
SBlastema = df.neighbors[0]
SBlastema = [SBlastema[i] for i in idx]
y = np.array(SBlastema)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SOther = df.neighbors[1]
SOther = [SOther[i] for i in idx]
y = np.array(SOther)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPES = df.neighbors[2]
SPES = [SPES[i] for i in idx]
y = np.array(SPES)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


SEarlyGlom = df.neighbors[3]
SEarlyGlom = [SEarlyGlom[i] for i in idx]
y = np.array(SEarlyGlom)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPodo = df.neighbors[5]
SPodo = [SPodo[i] for i in idx]
y = np.array(SPodo)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPT = df.neighbors[6]
SPT = [SPT[i] for i in idx]
y = np.array(SPT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show() 

SLOH = df.neighbors[7]
SLOH = [SLOH[i] for i in idx]
y = np.array(SLOH)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SDCT = df.neighbors[8]
SDCT = [SDCT[i] for i in idx]
y = np.array(SDCT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


# In[30]:


## 100 micron

for i in df.index:   
    type_array = (np.where(adata1_df.obs["NeighborHoodsubType"]==i)[0])
    ## find nearest neighbors
    connectivities_cellType = adata1_df.obsp["100_spatial_connectivities"][type_array,:] # make an matrix with just cell type i comprising rows
    sum = connectivities_cellType.sum()
    count = df['count'].loc[i]
    nn = []
    for j_index, j in enumerate(adata1_df.obs["cellType_SCANVI_2"].unique()):
        type2_array = (np.where(adata1_df.obs["cellType_SCANVI_2"]==j)[0])
        nn.append(connectivities_cellType[:,type2_array].sum()/(count))
        #nn.append(connectivities_cellType[:,type2_array].sum()/(sum))
        df.at[i,"neighbors"] = nn
        
# Blastema
idx = [0, 2, 1, 3]
mylabels = ["Nephron", "UB_CT", "Stroma", "Endothelium"]
SBlastema = df.neighbors[0]
SBlastema = [SBlastema[i] for i in idx]
y = np.array(SBlastema)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SOther = df.neighbors[1]
SOther = [SOther[i] for i in idx]
y = np.array(SOther)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPES = df.neighbors[2]
SPES = [SPES[i] for i in idx]
y = np.array(SPES)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


SEarlyGlom = df.neighbors[3]
SEarlyGlom = [SEarlyGlom[i] for i in idx]
y = np.array(SEarlyGlom)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPodo = df.neighbors[5]
SPodo = [SPodo[i] for i in idx]
y = np.array(SPodo)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SPT = df.neighbors[6]
SPT = [SPT[i] for i in idx]
y = np.array(SPT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show() 

SLOH = df.neighbors[7]
SLOH = [SLOH[i] for i in idx]
y = np.array(SLOH)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()

SDCT = df.neighbors[8]
SDCT = [SDCT[i] for i in idx]
y = np.array(SDCT)
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the pie chart
wedges, texts = ax.pie(y, labels=mylabels, wedgeprops={"linewidth": 1, "edgecolor": "white"}, colors=["#698CFF", "black", "#794B82", "#7AE031"])
# Set the facecolor of the figure and axes to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
# Remove the legend
ax.legend().remove()
# Remove text labels to avoid legend-like appearance
for text in texts:
    text.set_visible(False)
# Optionally save the figure with a transparent background
plt.savefig("transparent_pie_chart.png", transparent=True)
# Show the plot
plt.show()


# In[31]:


from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler


# In[32]:


print(adata1.obs["cellType_SCANVI"].value_counts())

adata1_df = adata1[adata1.obs["cellType_SCANVI"].isin(["Int","PT","PEC","PT","DCT","Podocyte"])]
adata1_df = adata1_df[adata1_df.obs["NeighborHoodsubType"].isin(["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"])]

# Assuming 'adata' is your AnnData object
markers = ["IGF2", "MEIS2","UNCX","SIX2","SIX1","LHX1","ENO1","JAG1","WT1","NPHS2"]

# Ensure markers list only includes valid gene names
markers = [gene for gene in markers if gene in adata.var_names]

# Convert .X to a dataframe for GOI
gene_expression_df = pd.DataFrame(
    adata1_df[:, markers].X.toarray() if issparse(adata.X) else adata1_df[:, markers].X,
    index=adata1_df.obs_names,
    columns=markers
)

gene_expression_df['NeighborHoodsubType'] = adata1_df.obs['NeighborHoodsubType'].values
mean_expression_per_cluster = gene_expression_df.groupby('NeighborHoodsubType').mean()

scaler = MinMaxScaler()
gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)

# Define the order
order = ["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"]

# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()


# In[ ]:





# In[33]:


adata1_df = adata1_df[adata1_df.obs["NeighborHoodsubType"].isin(["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"])]

# Assuming 'adata' is your AnnData object
markers = ['PseudoTime_SCVI','NPC_SCVI', 'Differentiated Cell_SCVI','Tubule_SCVI', 'Glomerular_SCVI','Podo_absorbtion_SCVI','G2M_score', 'S_score']

# Ensure markers list only includes valid gene names
gene_expression_df = adata1_df.obs[markers]

gene_expression_df['NeighborHoodsubType'] = adata1_df.obs['NeighborHoodsubType'].values
mean_expression_per_cluster = gene_expression_df.groupby('NeighborHoodsubType').mean()

scaler = MinMaxScaler()
gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)

# Define the order
order = ["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"]

# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()


# annotate neighboring cells for each neighborhood

# In[34]:


from scipy.sparse import csr_matrix
all_dfs = []
connectivity_matrix = csr_matrix(adata1.obsp["fine_spatial_connectivities"])
temp_df = pd.DataFrame(index=adata1.obs.index)

neighbor_cell_ids = []
for cell_index in range(connectivity_matrix.shape[0]):
    neighbor_indices = connectivity_matrix[cell_index, :].nonzero()[1]
    neighbor_ids = adata1.obs.index[neighbor_indices].tolist()
    neighbor_cell_ids.append(neighbor_ids)
temp_df["neighbor_ids"] = neighbor_cell_ids
all_dfs.append(temp_df)
df_neighbor_ids = pd.concat(all_dfs, axis=0)
df_neighbor_ids.sort_index(inplace=True)

df_neighbor_ids['NeighborHoodsubType']=adata1.obs['NeighborHoodsubType']
subset_df=df_neighbor_ids[df_neighbor_ids['NeighborHoodsubType'].isin(["PT","LOH","PEC","DCT","Podocyte"])]
subset_df

# DataFrame for the index and cluster_label
index_cluster_df = subset_df.reset_index().rename(columns={'index': 'cell_id'})[['cell_id', 'NeighborHoodsubType']]
# DataFrame for the neighbor_ids and cluster_label
neighbors_cluster_df = subset_df.reset_index(drop=True)[['neighbor_ids', 'NeighborHoodsubType']]
print(index_cluster_df.head())  # Preview the DataFrame with cell_id and cluster_label
print(neighbors_cluster_df.head())  # Preview the DataFrame with neighbor_ids and cluster_label
# Explode the 'neighbor_ids' column to have each neighbor_id in its own row
exploded_neighbors_cluster_df = neighbors_cluster_df.explode('neighbor_ids')
# Drop duplicates in the 'neighbor_ids' column
exploded_neighbors_cluster_df = exploded_neighbors_cluster_df.drop_duplicates(subset='neighbor_ids')
print(exploded_neighbors_cluster_df)  # Preview the exploded and deduplicated DataFrame
# Rename 'neighbor_ids' column to 'cell_id' in exploded_neighbors_cluster_df for consistency
exploded_neighbors_cluster_df = exploded_neighbors_cluster_df.rename(columns={'neighbor_ids': 'cell_id'})
# Concatenate the DataFrames vertically
combined_df = pd.concat([index_cluster_df, exploded_neighbors_cluster_df], ignore_index=True)
print(combined_df)  # Preview the combined DataFrame
combined_df.drop_duplicates(subset=['cell_id'], inplace=True)
combined_df
# Create a dictionary from 'cell_id' to 'cluster_labels'
cell_cluster_dict = dict(zip(combined_df['cell_id'], combined_df['NeighborHoodsubType']))
# Map the dictionary to a new column in your adata object
adata1.obs['neighborhood'] = adata1.obs_names.map(cell_cluster_dict).astype('category')
adata1.obs['neighborhood'] = adata1.obs['neighborhood'].cat.add_categories(["Blastema","NephrogenicZoneOther","PES","EarlyGlom"])

adata1.obs['NeighborHoodsubType'].fillna(adata1.obs['neighborhood'])

print(adata1.obs['neighborhood'].value_counts())
print(adata1.obs['NeighborHoodsubType'].value_counts())


# In[35]:


adata1_df = adata1[~adata1.obs["cellType_SCANVI"].isin(["Ureth","Immune Cells"])]
adata1_df = adata1_df[adata1_df.obs["NeighborHoodsubType"].isin(["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"])]

adata1_df.X = adata1_df.layers["counts"]

# Assuming 'adata' is your AnnData object
markers = ["IGF2", "MEIS2","UNCX","SIX2","SIX1","LHX1","ENO1","JAG1","WT1","NPHS2"]

# Ensure markers list only includes valid gene names
markers = [gene for gene in markers if gene in adata.var_names]

# Convert .X to a dataframe for GOI
gene_expression_df = pd.DataFrame(
    adata1_df[:, markers].X.toarray() if issparse(adata.X) else adata1_df[:, markers].X,
    index=adata1_df.obs_names,
    columns=markers
)

gene_expression_df['NeighborHoodsubType'] = adata1_df.obs['NeighborHoodsubType'].values
mean_expression_per_cluster = gene_expression_df.groupby('NeighborHoodsubType').mean()

scaler = MinMaxScaler()
gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)

# Define the order
order = ["Blastema","NephrogenicZoneOther","PES","EarlyGlom","Podocyte"]

# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()

