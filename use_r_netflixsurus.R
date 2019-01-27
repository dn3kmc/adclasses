# use_r_netflixsurus.R

library("RAD")

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)

# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

mydata$timestamp <- as.POSIXct(mydata$timestamp)

specificdata <- 
data.frame(mydata$timestamp,mydata$value)

# rpca assumes the data has a periodicity (called frequency) of 
# 7 unless another value is passed. 
# Your data must be divisible by 7 if you don't pass a value 
# for frequency when calling the function. 
# Devs recommend using frequency=1 if you don't have an 
# expectation of periodicity in your data.

# Convert to numerics
# freq
freq = as.numeric(myArgs[2])

cat(" freq: ")
cat(freq)

# autodiff: boolean, True -> use ADF to determine if differencing is
# needed to make ts stationary, default is True

# missing
autodiff = as.logical(myArgs[3])

cat(" autodiff: ")
cat(autodiff)

# forcediff: boolean, True -> always compute differences, default is False

forcediff = as.logical(myArgs[4])

cat(" forcediff: ")
cat(forcediff)

# scale: boolean, True -> normalize the ts to 0 mean and 1 variance, default is True

scale = as.logical(myArgs[5])

cat(" scale: ")
cat(scale)

# L.penalty, scalar -> thresholding for low rank approximation of ts,
# default is 1

if (myArgs[6] == "default") {
    lpenalty = 1
} else {
    lpenalty = as.numeric(myArgs[6])
}

cat(" L.penalty: ")
cat(lpenalty)

# s.penalty, scalar -> thresholding for separating noise and sparse outliers
# default is 1.4 / sqrt(max(frequency, ifelse(is.data.frame(X), nrow(X), length(X)) / frequency))

if (myArgs[7] == "default") {
    spenalty = 1.4 / sqrt(max(freq, ifelse(is.data.frame(mydata$value), nrow(mydata$value), length(mydata$value)) / freq))
} else {
    spenalty = as.numeric(myArgs[7])
}

cat(" S.penalty: ")
cat(spenalty)

rpca_res <- AnomalyDetection.rpca(mydata$value, frequency=freq, autodiff=autodiff, forcediff=forcediff, scale=scale, L.penalty=lpenalty, s.penalty=spenalty)

rpca_res_an <- subset(rpca_res, abs(S_transform) > 0)

# head(rpca_res_an$time)

write.csv(file="r_dfs/netflix_surus_outliers_index_1.csv",x=rpca_res_an$time)

# rpca_res_an$time will give you R indices of outliers
# warning! R is 1 indexed not 0 indexed!
# so if R thinks t is an outlier, it is actually t-1 in Python

cat("True")