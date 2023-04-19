Raw <-read.csv("test.csv",comment.char = "#",na.strings = "无此项",check.names = F)
Raw <- Raw[!is.na(Raw$Cost_National) ,] # 去除空白信息列

apply(Raw,2,is.numeric)
Raw.Num <- Raw[,-c(1,2,7:11)]
apply(Raw.Num,2,is.numeric)

Raw.Num<-Raw.Num[,-42] #delete order
apply(Raw.Num,2,function(x){return(sum(is.na(x)))})


library(randomForest)

#Raw.Num  是巫老师即将分享的数据
###################
######仅利用2018年数据
###################

Tem <-Raw.Num[ Raw.Num$Year %in% c( 2018) ,-grep("_Score",colnames(Raw.Num))]
Tem <- Tem[,-c(1,ncol(Tem))] #year & ID
Tem<-Tem[,apply(Tem,2,function(x){return(sum(is.na(x)))})<=0 ]

write.csv(Tem,"tables/Input.randomForest.2018.csv",row.names = F)


set.seed(123)
Samples <- sample(1:nrow(Tem),ceiling(nrow(Tem)*0.7))

Train.tem <- Tem[ Samples, ]

set.seed(123)
otu_train.forest <- randomForest(Score~., data = Train.tem, importance = TRUE,ntree=500)
otu_train.forest

plant_predict <- predict(otu_train.forest, Train.tem)


pdf("2018.sample.RF.Validation.pdf",width = 9,height = 6)
par(mfrow=c(1,2),mar = (c(4.2,5,2,0)+ 0.5))
plot(Train.tem$Score, plant_predict, main = 'Train',ylim = c(0,100),xlim = c(0,100),
     xlab = 'Score', ylab = 'Predict',cex.lab=1.4)
abline(1, 1,col="red")
text(15,85,paste0("R=",round(cor(Train.tem$Score, plant_predict),2)),cex=1.2,col="blue")

plant_predict <- predict(otu_train.forest,  Tem[ -(Samples), ])


plot(Tem$Score[ -(Samples) ], plant_predict, main = 'Test',ylim = c(0,100),xlim = c(0,100),
     xlab = 'Score', ylab = 'Predict',cex.lab=1.4)
text(15,85,paste0("R=",round(cor(Tem$Score[ -c(Samples) ], plant_predict),2)),cex=1.2,col="blue")
abline(1, 1,col="red")
dev.off()


importance_otu <- otu_train.forest$importance
head(importance_otu)

#或者使用函数 importance()
importance_otu <- data.frame(importance(otu_train.forest), check.names = FALSE)
head(importance_otu)

pdf("2018.sample.RF.Importance.pdf",width = 9,height = 6)
#作图展示 重要的 OTUs
varImpPlot(otu_train.forest, n.var = min(30, nrow(otu_train.forest$importance)),
           main = 'variable importance')
dev.off()
# end edit @ 2023-04-19