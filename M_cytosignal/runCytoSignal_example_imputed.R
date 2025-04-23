library(Matrix)
library(cytosignal)
library(Seurat)

## The RDS file will be loaded into a ready-to-use object
dge <- readRDS("/home/levinsj/spatial/adata/Export_to_R_files/FK1_Matrix_imputed.rds")

## The cluster annotation need to be presented as a factor object
cluster <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/FK1_clusters.csv")
cluster <- factor(cluster$cellType_SCANVI)
names(cluster) <- colnames(dge)

## The spatial coordinates need to be presented as a matrix object
spatial <- as.matrix(read.csv("/home/levinsj/spatial/adata/Export_to_R_files/FK1_locations.csv", row.names = 1))
## Please make sure that the dimension names are lower case "x" and "y"
colnames(spatial) <- c("x", "y")

##
cs <- createCytoSignal(raw.data = dge, cells.loc = spatial, clusters = cluster)

cs <- addIntrDB(cs, g_to_u, db.diff, db.cont, inter.index)

cs <- removeLowQuality(cs, counts.thresh = 0, gene.thresh = 0)
cs <- changeUniprot(cs)

print(cs)

cs <- inferEpsParams(cs, scale.factor = 0.12)
print(cs@parameters$r.diffuse.scale)
cs <- findNN(cs)
cs <- imputeLR(cs)

cs <- inferIntrScore(cs, recep.smooth = FALSE)
cs <- inferSignif(cs, p.value = 0.05, reads.thresh = 100, sig.thresh = 50)
cs <- rankIntrSpatialVar(cs)

difIntrs <- showIntr(cs, slot.use = "GauEps-Raw", signif.use = "result.spx", return.name = TRUE)
contIntrs <- showIntr(cs, slot.use = "DT-Raw", signif.use = "result.spx", return.name = TRUE)

write.csv(difIntrs, "/home/levinsj/spatial/adata/Export_to_R_files/imputedIntrs_ligand_FK1_raw.csv")
write.csv(contIntrs, "/home/levinsj/spatial/adata/Export_to_R_files/imputedIntrs_contact_FK1_raw.csv")

saveRDS(cs, file = "/home/levinsj/spatial/adata/Export_to_R_files/FK1_imputedCytoSignal_raw.rds")

