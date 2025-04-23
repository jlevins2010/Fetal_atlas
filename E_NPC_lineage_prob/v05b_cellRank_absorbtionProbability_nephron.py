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


# In[3]:


adata = scv.read("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineage_40_30_clean.h5ad")

scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType", palette = "Dark2")


# # Generate kernel

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


# # Create Estimator

# In[5]:


from cellrank.tl.estimators import GPCCA

g = GPCCA(cbk)
print(g)

g.compute_schur(n_components=7)
g.plot_spectrum()

g.compute_macrostates(n_states=7, cluster_key="cellType", n_cells = 400)
#considere removing clusters, changing to cellType
g.plot_macrostates()
g.plot_macrostates(same_plot=False)
g.plot_macrostates(discrete=True)


# # Set initial states as NPC

# In[6]:


initial_states = (g.macrostates == 'NPC')

adata.obs["initial_states_probs"] = g.macrostates_memberships["NPC"]
adata.obs["initial_states"] = initial_states


# # Set terminal states

# In[7]:


macroStateSize = 500

adata.obs["NPC"] = g.macrostates_memberships["NPC"]
NPC_macroState = adata.obs.nlargest(macroStateSize, 'NPC').index.tolist()

adata.obs["Podocyte"] = g.macrostates_memberships["Podocyte"]
Podo_macroState = adata.obs.nlargest(macroStateSize, 'Podocyte').index.tolist()

adata.obs["PT"] = g.macrostates_memberships["PT_1"]
PT_macroState = adata.obs.nlargest(macroStateSize, 'PT').index.tolist()

adata.obs["DCT"] = g.macrostates_memberships["DCT"]
DCT_macroState = adata.obs.nlargest(macroStateSize, 'DCT').index.tolist()

adata.obs["LOH"] = g.macrostates_memberships["LOH"]
LOH_macroState = adata.obs.nlargest(macroStateSize, 'LOH').index.tolist()


# In[8]:


g.set_terminal_states({"PT": PT_macroState, "LOH": LOH_macroState, "DCT": DCT_macroState, "Podocyte": Podo_macroState})
g.compute_absorption_probabilities()
g.plot_absorption_probabilities()

g.plot_absorption_probabilities(same_plot = False)

g.plot_coarse_T(text_kwargs={"fontsize": 10})

adata.uns['iroot'] = np.flatnonzero(adata.obs['initial_states'] == True)[0]
sc.tl.dpt(adata)

scv.tl.recover_latent_time(
    adata, root_key="initial_states_probs", end_key="terminal_states_probs"
)


# In[9]:


g.plot_absorption_probabilities(legend_loc = 'right margin', same_plot = False)


# In[10]:


scv.tl.paga(
    adata,
    groups="leiden_orig",
    root_key="initial_states_probs",
    end_key="terminal_states_probs",
    use_time_prior="dpt_pseudotime",
)

sc.set_figure_params(figsize=(10, 10))

cr.pl.cluster_fates(
    adata,
    mode="paga_pie",
    cluster_key="leiden_orig",
    basis="umap",
    legend_kwargs={"loc": "bottom left"},
    legend_loc="on data",
    node_size_scale=5,
    edge_width_scale=1,
    max_edge_width=4,
    title="directed PAGA",
)


# In[11]:


df = pd.DataFrame({'PseudoTime': adata.obs["dpt_pseudotime"], 'LatentTime': adata.obs["latent_time"]})
df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/pseudoTime.csv', index=False)


# In[12]:


model = cr.models.GAM(adata)


# In[13]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["MEIS2"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)



# In[14]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["UNCX"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[15]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["JAG1"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[16]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["LHX1"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[17]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["LHX1"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[18]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["APOE"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[19]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["WT1"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[20]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["NPHS2"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[21]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["DAB2"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# In[22]:


cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["IRX1"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)
cr.pl.gene_trends(
    adata,
    model=model,
    data_key="counts",
    genes=["IRX2"],
    time_key="dpt_pseudotime",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    same_plot=True
)


# cr.pl.gene_trends(
#     adata,
#     model=model,
#     data_key="counts",
#     genes=["UMOD"],
#     time_key="dpt_pseudotime",
#     hide_cells=True,
#     weight_threshold=(1e-3, 1e-3),
#     same_plot=True
# )
# cr.pl.gene_trends(
#     adata,
#     model=model,
#     data_key="counts",
#     genes=["SLC12A1"],
#     time_key="dpt_pseudotime",
#     hide_cells=True,
#     weight_threshold=(1e-3, 1e-3),
#     same_plot=True
# )
# cr.pl.gene_trends(
#     adata,
#     model=model,
#     data_key="counts",
#     genes=["WNK1"],
#     time_key="dpt_pseudotime",
#     hide_cells=True,
#     weight_threshold=(1e-3, 1e-3),
#     same_plot=True
# )
