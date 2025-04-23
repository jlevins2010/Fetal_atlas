library(Matrix)
library(cytosignal)
library(Seurat)
library(dplyr)


### load RDS of processed cytoSignal 
cs <- readRDS("/home/levinsj/spatial/adata/Export_to_R_files/FK1_imputedCytoSignal_raw.rds")

## Get pair indexes
ligs <- unique(names(cs@intr.valid[["diff_dep"]]$ligand))
#ligs <- unique(names(cs@intr.valid[["diff_dep"]]$combined))

print(ligs[1:10])


score.obj <- cs@lrscore[["GauEps-Raw"]]
### interaction database
intr.db <- cs@intr.valid[[score.obj@intr.slot]]
lig.slot <- score.obj@lig.slot
dge.lig <- cs@imputation[[lig.slot]]@imp.data
lig_matrix <- matrix(nrow = ncol(dge.lig), ncol = 0)
ligNames <- list()
suffix <- "_lig"

print(rownames(dge.lig))

print(length(ligs))
ligs <- ligs[ligs %in% rownames(dge.lig)]
print(length(ligs))

#for (i in ligs){
#    ligand.name = cs@intr.valid$intr.index[i, 4]
#    if (ligand.name == ""){ # if complex
#        ligand.name = cs@intr.valid$intr.index[i, 2]}
#    ligand.name = gsub("_HUMAN", "", ligand.name)
#    ligand.name <- paste0(ligand.name, suffix)
#    ## -- good up to here...
#    ligNames <- c(ligNames, ligand.name[1])
#    if (i %in% names(intr.db[[2]][intr.db[[2]]])) {
#        ligands <- dge.lig[names(intr.db[[2]][intr.db[[2]] == i] ), ]
#    } else {
#        ligands <- matrix(0, nrow = 1, ncol = ncol(dge.lig))
#    }
#    if (!is.null(nrow(ligands))){ligands = Matrix::colSums(ligands)}
#    lig_matrix <- cbind(lig_matrix, ligands)
#}

#ligNames <- unlist(ligNames)
#colnames(lig_matrix) <- ligNames
#print(dim(lig_matrix))#

#ligs <- unique(cs@intr.valid[["cont_dep"]]$ligands)
#print(length(ligs))
#ligs <- ligs[names(ligs) %in% names(intr.db[[2]][intr.db[[2]]] )]
#print(length(ligs))
                                          
#score.obj <- cs@lrscore[["DT-Raw"]]
## interaction database
#intr.db <- cs@intr.valid[[score.obj@intr.slot]]
#lig.slot <- score.obj@lig.slot
#dge.lig <- cs@imputation[[lig.slot]]@imp.data
#con_matrix <- matrix(nrow = ncol(dge.lig), ncol = 0)
#ligNames <- list()
#suffix <- "_contact"#

#for (i in ligs){
#    ligand.name = cs@intr.valid$intr.index[i, 4]
#    if (ligand.name == ""){ # if complex
#        ligand.name = cs@intr.valid$intr.index[i, 2]}
#    ligand.name = gsub("_HUMAN", "", ligand.name)
#    ligand.name <- paste0(ligand.name, suffix)
#    ligNames <- c(ligNames, ligand.name[1])
#    if (i %in% names(intr.db[[2]][intr.db[[2]]])) {
#        ligands <- dge.lig[names(intr.db[[2]][intr.db[[2]] == i] ), ]
#    } else {
#        ligands <- matrix(0, nrow = 1, ncol = ncol(dge.lig))
#    }
#    if (!is.null(nrow(ligands))){ligands = Matrix::colSums(ligands)}
#    con_matrix <- cbind(con_matrix, ligands)
#}

#ligNames <- unlist(ligNames)
#colnames(con_matrix) <- ligNames
#print(dim(con_matrix))

#dense_mat <- t(cbind(lig_matrix, con_matrix))
#print(dim(dense_mat))

#print(dense_mat[1:10,1:10])
#write.csv(dense_mat, "/home/levinsj/spatial/adata/Export_to_R_files/FK1_all_ligand_scores_raw.csv")