library(Matrix)
library(cytosignal)
library(Seurat)
library(dplyr)


### load RDS of processed cytoSignal 
ItrFK1 <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/imputedIntrs_FK1.csv")
ItrFK2 <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/imputedIntrs_FK4.csv")
ItrFK3 <- read.csv("/home/levinsj/spatial/adata/Export_to_R_files/imputedIntrs_HK3524.csv")

print(dim(ItrFK1))
print(dim(ItrFK2))
print(dim(ItrFK3))

combinedInteractions <- rbind(ItrFK1, ItrFK2, ItrFK3)

print(dim(combinedInteractions))
print(combinedInteractions)

colnames(combinedInteractions) <- c("X", "interaction")

combinedInteractions <- distinct(combinedInteractions, X)

print(dim(combinedInteractions))
print(combinedInteractions)


write.csv(combinedInteractions, "/home/levinsj/spatial/adata/Export_to_R_files/sharedInts.csv")
