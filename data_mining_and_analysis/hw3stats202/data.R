setwd("/Users/tranlm/Google Drive/homework/hw3/data")

#Ch7 Q8
data(Auto)
write.csv(Auto, file='./Auto.csv', row.names=FALSE)

# Ch9 Q8
require(ISLR)
data(OJ)
write.csv(OJ, file='./OJ.csv', row.names=FALSE)

# Chxxx
data(College)
write.csv(College, file='./College.csv', row.names=TRUE)
