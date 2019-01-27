# use_r_twitterad.R

# import libraries
library(AnomalyDetection)

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)
# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

# max_anoms
max_anoms = as.numeric(myArgs[2])
# direction
direction = myArgs[3]
# alpha
alpha = as.numeric(myArgs[4])
# period
period = as.numeric(myArgs[5])

res = AnomalyDetectionVec(as.vector(mydata["value"]), max_anoms=max_anoms, alpha=alpha, period=period,direction=direction, plot=FALSE)

write.csv(file="r_dfs/twitter_index_anomalies.csv",x=res$anoms$index)

cat("True")