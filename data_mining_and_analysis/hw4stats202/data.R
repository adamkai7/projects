setwd("/Users/tranlm/Google Drive/homework/hw4/data")

data(Carseats)
write.csv(Carseats, file='./Carseats.csv', row.names=FALSE)

data(Hitters)
write.csv(Hitters, file='./Hitters.csv', row.names=TRUE)

data(Caravan)
write.csv(Caravan, file='./Caravan.csv', row.names=FALSE)

