# use_r_stl.R

# import libraries
library("stlplus")
library("imputeTS")

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)

# cat(if (myArgs[3]=="periodic") { "hey"} else {"nay"})

# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

cat(length(mydata$value))


# convert BeginTime column to time 
mydata$timestamp <- as.POSIXct(mydata$timestamp)

# Convert to numerics
# n.p.
np = as.numeric(myArgs[2])
# s.window
if (myArgs[3]=="periodic") {
    swindow="periodic"
} else {
    swindow=as.numeric(myArgs[3])
}
# s.degree
sdegree = as.numeric(myArgs[4])
# t.window
twindow = as.numeric(myArgs[5])
# t.degree
tdegree = as.numeric(myArgs[6])
# inner
inner = as.numeric(myArgs[7])
# outer
outer = as.numeric(myArgs[8])

# missing
missing = as.logical(myArgs[9])
# fill_option
fill_option = myArgs[10]
# name to give output csv
name = myArgs[11]

# cat(length(mydata$value))

# use stl
stltest <- stlplus(x=mydata$value, t=mydata$timestamp, n.p=np, s.window=swindow, s.degree = sdegree, t.window = twindow, t.degree = tdegree, inner = inner, outer = outer)

stltest$data$timestamp <- mydata$timestamp

if (missing) {
    cat("  missing. will fill. ")
    remainder_filled = na.interpolation(stltest$data$remainder,option=fill_option)
    stltest$data$remainder_filled <- remainder_filled
} else {
    cat(" no missing ")
}

filename=paste("r_dfs/df_to_csv_stl_", name, ".csv",sep="")
write.csv(file=filename,x=stltest$data)

cat("True")
