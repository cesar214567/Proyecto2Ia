df <- read.csv("/home/enrique/Documentos/IA/Proyecto2Ia/dbknn.csv", 
               sep = ",", header = FALSE)

df <- df[c(1:36)]

d.pca<-princomp(df)
summary(d.pca)
nd <- d.pca$scores[,c(1:12)]

newdf <- df[c(1:12)]


write.csv(newdf, "/home/enrique/Documentos/IA/Proyecto2Ia/newknndb.csv",row.names = FALSE,col.names = NA)


