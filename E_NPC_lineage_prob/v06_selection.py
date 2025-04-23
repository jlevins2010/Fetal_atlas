#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
import anndata
import scanpy as sc
import scvelo as scv
import cellrank as cr


# In[2]:


scv.set_figure_params('scvelo')


# adata_nep = scv.read('/home/levinsj/Fetal_dir/Velocyto/02_mergedSCV/allFetal.h5ad', cache=True)
# ldata = scv.read('/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephrogenicOnly_annotated.h5ad')
# adata_nep = scv.utils.merge(adata_nep, ldata)
# 
# scv.pp.filter_and_normalize(adata_nep, min_shared_counts=20, n_top_genes=2000)
# scv.pp.moments(adata_nep, n_pcs=40, n_neighbors=30)
# scv.tl.recover_dynamics(adata_nep)
# scv.tl.velocity(adata_nep, mode="dynamical", n_jobs=8)
# scv.tl.velocity_graph(adata_nep)
# 
# scv.pl.velocity_embedding_stream(adata_nep, basis='umap', color = "leiden")
# 
# adata_nep.write_h5ad(filename = "/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineage.h5ad")
# 
# #cr.tl.terminal_states(adata_nep, cluster_key="leiden", weight_connectivities=0.2)
# #cr.pl.terminal_states(adata_nep)
# 
# 
# 

# In[3]:


adata = scv.read("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineage_40_30_clean.h5ad")


scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType")


# In[4]:


from cellrank.tl.kernels import VelocityKernel

vk = VelocityKernel(adata)
vk.compute_transition_matrix()
print(vk)

from cellrank.tl.kernels import ConnectivityKernel

#using knn we will generate a connectivity kernel, this is from the knn graph based on PCs from the harmony calculation
print(adata)

ck = ConnectivityKernel(adata).compute_transition_matrix()
print(ck)

# weighting of velocity vs connectivity
cbk = 0.5 * vk + 0.5 * ck
print(cbk)


# In[5]:


from cellrank.tl.estimators import GPCCA

g = GPCCA(cbk)
print(g)

g.compute_schur(n_components=7)
g.plot_spectrum()

g.compute_macrostates(n_states=7, cluster_key="cellType")
#considere removing clusters, changing to cellType
g.plot_macrostates()
g.plot_macrostates(same_plot=False)
g.plot_macrostates(discrete=True)


# In[6]:


initial_states = (g.macrostates == 'NPC')

adata.obs["initial_states_probs"] = g.macrostates_memberships["NPC"]
adata.obs["initial_states"] = initial_states


# In[7]:


adata.uns['iroot'] = np.flatnonzero(adata.obs['initial_states'] == True)[0]
sc.tl.dpt(adata)


# In[8]:


macroStateSize = 500

adata.obs["NPC"] = g.macrostates_memberships["NPC"]
NPC_macroState = adata.obs.nlargest(macroStateSize, 'NPC').index.tolist()

adata.obs["Podocyte"] = g.macrostates_memberships["Podocyte"]
Podo_macroState = adata.obs.nlargest(macroStateSize, 'Podocyte').index.tolist()

adata.obs["PT"] = g.macrostates_memberships["PT_1"]
PT_macroState = adata.obs.nlargest(macroStateSize, 'PT').index.tolist()

adata.obs["PT2"] = g.macrostates_memberships["PT_2"]
PT2_macroState = adata.obs.nlargest(macroStateSize, 'PT2').index.tolist()

adata.obs["DCT"] = g.macrostates_memberships["DCT"]
DCT_macroState = adata.obs.nlargest(macroStateSize, 'DCT').index.tolist()

adata.obs["LOH"] = g.macrostates_memberships["LOH"]
LOH_macroState = adata.obs.nlargest(macroStateSize, 'LOH').index.tolist()


# In[9]:


g.set_terminal_states({"Tubule": PT_macroState + DCT_macroState + LOH_macroState, "Renal Corpuscle": Podo_macroState})
#g.set_terminal_states({"Tubule": adata[adata.obs['cellType'].isin(["PT", 'LOH', 'DCT'])].obs_names, "Glomerular": adata[adata.obs['cellType'].isin(['Podocyte','PEC'])].obs_names})

g.compute_absorption_probabilities()
g.plot_absorption_probabilities()

df = pd.DataFrame(g.absorption_probabilities, columns = g.absorption_probabilities.names)

df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/selection_absorbtion_probabilities.csv', index=False)

g.compute_lineage_drivers(return_drivers=True)


# In[10]:


df = g.compute_lineage_drivers(return_drivers=True)
df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineageDrivers_selection.csv')  
df = pd.read_csv("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineageDrivers_selection.csv")
TFs = pd.read_csv("/home/levinsj/Applications/TFs_hg38.txt", header=None)
TF_df = df[df["Unnamed: 0"].isin(TFs[0])]
TF_df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineageDrivers_selection_TFs.csv') 

