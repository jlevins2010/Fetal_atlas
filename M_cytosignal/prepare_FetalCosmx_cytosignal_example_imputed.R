library(Matrix)
library(cytosignal)
library(Seurat)

## Prepare MetaData
metaData <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/FK1_obs_names.csv")
metaData <- metaData[which(metaData$sample == "1"), ]
write.table(metaData$X, "/home/levinsj/spatial/adata/Export_to_R_files/FK1_barcodes.csv", sep = ',', row.names = FALSE, col.names = FALSE, quote=FALSE)

## Prepare gene list
var_names <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/FK1_var_names.csv")
allGenes <- var_names$X
CosMxGenes <- var_names$X[which(var_names$CosMx == "True")]

### Write to table
write.table(allGenes, "/home/levinsj/spatial/adata/Export_to_R_files/allGenes.csv", sep = ',', row.names = FALSE, col.names = FALSE, quote=FALSE)
write.table(CosMxGenes, "/home/levinsj/spatial/adata/Export_to_R_files/CosMxGenes.csv", sep = ',', row.names = FALSE, col.names = FALSE, quote=FALSE)

### Write locations and clusters
clusters <- subset(metaData, select=c("cellType_SCANVI"))
locations <- subset(metaData, select=c("CenterX_global_px", "CenterY_global_px"))

colnames(locations) <- c("X","Y")
rownames(locations) <- (metaData$X)
rownames(clusters) <- (metaData$X)

print(length(metaData$X))

write.table(locations, "/home/levinsj/spatial/adata/Export_to_R_files/FK1_locations.csv", sep = ',', row.names = TRUE, col.names = TRUE, quote=FALSE)
write.table(clusters, "/home/levinsj/spatial/adata/Export_to_R_files/FK1_clusters.csv", sep = ',', row.names = TRUE, col.names = TRUE, quote=FALSE)


### Write Counts Matrix
countsMatrix <- ReadMtx(mtx = "/home/levinsj/spatial/adata/Export_to_R_files/FK1_imputed.mtx", features = "/home/levinsj/spatial/adata/Export_to_R_files/allGenes.csv", cells = "/home/levinsj/spatial/adata/Export_to_R_files/FK1_barcodes.csv", feature.column = 1, cell.column = 1, mtx.transpose = TRUE)

saveRDS(countsMatrix, file = "/home/levinsj/spatial/adata/Export_to_R_files/FK1_Matrix_imputed.rds")

