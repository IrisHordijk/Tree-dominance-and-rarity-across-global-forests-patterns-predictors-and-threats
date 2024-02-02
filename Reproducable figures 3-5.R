###Reproducable figures

###Load packages
library(datasets)
library(data.table)
library(xlsx)
library(feather)
library(tidyr)
library(tidyverse)
library(stringr)
library(car)
library(dunn.test)
library(varImp)
library(caret)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(plotrix)
library(dplyr)

###Figure 3
DataFig3 <- fread("DataFig3.csv")
DataFig3.1 <- fread("DataFig3.1.csv")

###Global graph
GraphDom_rar <- ggplot(DataFig3, aes(x=BA_perc_sp, y=Rarity)) + geom_point(alpha=0.01, colour="grey") + geom_smooth(aes(color=factor(Biome)), method="lm", linewidth=2) + scale_colour_manual(breaks=c("1", "2", "3", "4", "5", "6"), values=c("#473901", "#025914", "#11910a", "#0c1bc4",  "#0e94ed", "#6edefa")) + theme_classic()+  ggtitle("Dominance-Rarity")+ ylab("Rarity (%)") + xlab("Dominance (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position = "none",
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15))

GraphDom_S <- ggplot(DataFig3.1, aes(x=log(S), y=BA_perc_sp)) + geom_point(alpha=0.01, colour="grey") + geom_smooth(aes(color=factor(Biome)), method="lm", linewidth=2) + scale_colour_manual(breaks=c("1", "2", "3", "4", "5", "6"), values=c("#473901", "#025914", "#11910a", "#0c1bc4",  "#0e94ed", "#6edefa")) + theme_classic()+  ggtitle("Species richness-Dominance")+ ylab("Dominance (%)") + xlab("log(Species richness)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position = "none",
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15))

GraphRare_S <- ggplot(DataFig3, aes(x=log(S), y=Rarity)) + geom_point(alpha=0.1, colour="grey") + geom_smooth(aes(color=factor(Biome)), method="lm", linewidth=2) + scale_colour_manual(breaks=c("1", "2", "3", "4", "5", "6"), values=c("#473901", "#025914", "#11910a", "#0c1bc4",  "#0e94ed", "#6edefa")) + theme_classic()+  ggtitle("Species richness-Rarity")+ ylab("Rarity (%)") + xlab("log(Species richness)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position = "none",
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15))

Fig3 <- grid.arrange(GraphDom_rar, GraphDom_S, GraphRare_S, ncol=3, nrow=1) #See for legend the figure in the main text

###Figure 4
DataFig4A <- fread("DataFig4A.csv")

DataFig4A$Category <- factor(DataFig4A$Category, levels=c("Elevation", "Tree density", "Human impact", "Forest age", "Climate", "Soil"))
Graph_VarImportance <- ggplot(data = DataFig4A, aes(y = Perc, x = Topic, fill = factor(Category))) + 
  geom_bar(stat="identity", position = "stack") + scale_fill_manual(values=c("#000000", "#828181", "#6e60bd", "#76bd60", "#ebd934", "#eb9234")) + 
  theme_classic()+  ggtitle("Global forests")+ ylab("Variable importance (%)")+
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position = "bottom",
        legend.text=element_text(size=15),
        legend.title = element_blank(), 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.ticks.x=element_blank()) +
  guides(fill=guide_legend(ncol=2, nrow=3,byrow=TRUE))


DataFig4B <- fread("DataFig4B.csv")
  
GraphAgeDLoess <- ggplot(DataFig4B, aes(ForestAge_rounded, Dominance_rounded)) + 
  geom_point(alpha=0.1, color="#76bd60") + annotate("text", label = "r^2 == 0.28", parse=TRUE, x = 90, y = 95, size = 5, colour = "black") +
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black", span=1) + xlim(c(0,100)) +  ylim(c(0,100)) + theme_classic() +
  ylab("Dominance") + xlab("Forest Age (yr)") + ggtitle("Forest age") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

GraphClimateDLoess <- ggplot(DataFig4B, aes(Precipitation_rounded, Dominance_rounded)) + 
  geom_point(alpha=0.1, color="#ebd934") + annotate("text", label = "r^2 == 0.28", parse=TRUE, x = 2700, y = 95, size = 5, colour = "black") + 
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black") + xlim(c(0,3000)) + theme_classic() + 
  ylab("Dominance") + xlab("Annual Precipitation (mm)") + ggtitle("Climate") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

GraphSoilDLoess <- ggplot(DataFig4B, aes(Sand_rounded, Dominance_rounded)) + 
  geom_point(alpha=0.1, color="#eb9234") + annotate("text", label = "r^2 == 0.08", parse=TRUE, x = 65, y = 95, size = 5, colour = "black") +
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black", span=0.8) + xlim(c(10,70))+ theme_classic() + 
  ylab("Dominance") + xlab("Sand content (%)") + ggtitle("Soil") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

GraphAgeRLoess <- ggplot(DataFig4B, aes(ForestAge_rounded, Rarity_rounded)) + 
  geom_point(alpha=0.1, color="#76bd60") + annotate("text", label = "r^2 == 0.06", parse=TRUE, x = 90, y = 95, size = 5, colour = "black") +
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black", span=1) + xlim(c(0,100)) + ylim(c(0,100)) + theme_classic() +
  ylab("Rarity") + xlab("Forest Age (yr)") + ggtitle("Forest age") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

GraphClimateRLoess <- ggplot(DataFig4B, aes(Precipitation_rounded, Rarity_rounded)) + 
  geom_point(alpha=0.1, color="#ebd934") + annotate("text", label = "r^2 == 0.001", parse=TRUE, x = 2700, y = 95, size = 5, colour = "black") +
  scale_colour_manual(breaks=c("1", "2", "3", "4", "5", "6"), values=c("#6edefa", "#0e94ed", "#0c1bc4", "#11910a", "#025914", "#473901")) +
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black", span=1) + xlim(c(0,3000)) + ylim(c(0,100)) + theme_classic() + #ylim(c(-2,2)) + 
  ylab("Rarity") + xlab("Annual Precipitation (mm)") + ggtitle("Climate") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

GraphSoilRLoess <- ggplot(DataFig4B, aes(Sand_rounded, Rarity_rounded)) + #, colour=factor(Biome)
  geom_point(alpha=0.1, color="#eb9234") + annotate("text", label = "r^2 == 0.01", parse=TRUE, x = 65, y = 95, size = 5, colour = "black") +
  geom_smooth(aes(group=1), method = "glm", method.args = list(family=gaussian(link="log")), colour="black", span=1) + xlim(c(10,70))+ ylim(c(0,100)) + theme_classic() + 
  ylab("Rarity") + xlab("Sand content (%)") + ggtitle("Soil") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position="none",
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 15))

layout=rbind(c(1,2,5), c(1,3,6), c(1,4,7))
Fig4 <- grid.arrange(Graph_VarImportance, GraphAgeDLoess,GraphClimateDLoess, GraphSoilDLoess,GraphAgeRLoess,GraphClimateRLoess,GraphSoilRLoess, layout_matrix=layout, nrow=3)

###Figure 5

DataFig5A <- fread("DataFig5A.csv")

BGCI_Perc <-  ggplot(data = DataFig5A, aes(y = Perc, x = factor(DominanceCategory), fill=factor(InterpretedAssessment))) + 
  geom_bar(stat="identity", position = "stack") + scale_fill_manual(values=c ("#850000", "#e33030", "#ffa8a8")) + #ylim(c(0,1)) + #"#a81616",
  ylab("Species globally (%)") + ggtitle('Conservation status') + scale_x_discrete(breaks=c("dominant", "rare"),labels= c("Dominant sp", "Rare sp")) + #scale_y_continuous(labels = function(l) {trans = l * 100 }) + 
  geom_label(aes(x = DominanceCategory, y = Perc, label = InterpretedAssessment, fontface = "bold"), position = position_stack(vjust = .5), size = 5, colour = "white", show.legend = FALSE) +
  theme_classic() + theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
                          legend.position = "none",
                          axis.text.x = element_text(size = 14, face = "bold", color ="black"),
                          axis.title.x = element_blank(),
                          axis.title.y = element_text(size = 16, face = "bold"),
                          axis.text.y = element_text(size = 14, face = "bold")) 

DataFig5B <- fread("DataFig5B.csv")

Global_PopulationTrend <-  ggplot(data = DataFig5B, aes(y = Perc, x = factor(DominanceCategory), fill=populationTrend)) + 
  geom_bar(stat="identity", position = "stack") + scale_fill_manual(values=c ("#b5b5b5", "#7d7d7d", "#424242", "#030303")) + #ylim(c(0,1)) +
  ylab("Threatened species globally (%)") + ggtitle('Population trend') + scale_x_discrete(breaks=c("dominant", "rare"),labels= c("Dominant sp", "Rare sp")) + 
  geom_label(aes(x = DominanceCategory, y = Perc, label = populationTrend, fontface = "bold"), position = position_stack(vjust = 0.3), size = 5, colour = "white", show.legend = FALSE) +
  theme_classic() + theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
                          legend.position = "none",
                          axis.text.x = element_text(size = 14, face = "bold", color ="black"),
                          axis.title.x = element_blank(),
                          axis.title.y = element_text(size = 16, face = "bold"),
                          axis.text.y = element_text(size = 14, face = "bold"))

DataFig5C <- fread("DataFig5C.csv")

Graph_endemics <-  ggplot(data = DataFig5C, aes(y = Perc, x = factor(DominanceCategory))) + geom_bar(stat="identity", fill="#4745bf") + 
  ylab("Threatened species (%)") + ggtitle('Endemism') + scale_x_discrete(breaks=c("dominant", "rare"),labels= c("Dominant sp", "Rare sp")) + 
  theme_classic() + theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
                          legend.position = "none",
                          axis.text.x = element_text(size = 14, face = "bold", color ="black"),
                          axis.title.x = element_blank(),
                          axis.title.y = element_text(size = 16, face = "bold"),
                          axis.text.y = element_text(size = 14, face = "bold"))

DataFig5D <- fread("DataFig5D.csv")

EOO_BGCI_Graph_Global <- ggplot(data=DataFig5D, aes(y=EOO, x=DominanceCategory, colour=DominanceCategory)) + geom_boxplot(fill="#a545bf", colour="black") +
  scale_y_continuous(labels = function(l) {trans = l / 100000 }, limits=c(0,15000000)) + 
  theme_classic() + ggtitle("Threatened species range") +  ylab("EOO (1000 ha)") + scale_x_discrete(breaks=c("dominant", "rare"),labels= c("Dominant sp", "Rare sp")) + 
  theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        legend.position = "none",
        axis.text.x = element_text(size = 14, face = "bold", color ="black"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.text.y = element_text(size = 14, face = "bold"))

margin = theme(plot.margin = unit(c(0.25,0.25,0.25,0.25), "cm"))
Fig5 <- grid.arrange(grobs = list(BGCI_Perc, Global_PopulationTrend,  Graph_endemics, EOO_BGCI_Graph_Global), nrow=2, ncol=2)

