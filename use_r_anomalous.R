# use_r_anomalous.R

# import libraries
library("anomalous")

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)
# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

# number of timeseries
num_ts = as.numeric(myArgs[2])
# upper limit on number of anomalies
upper = as.numeric(myArgs[3])

a <- matrix(mydata$value,ncol=num_ts)
z <- ts(a)

y <- tsmeasures(z)

write.csv(file="r_dfs/anomalous_index_anomalies.csv",x=anomaly(y,n=upper)$index)

cat("True")