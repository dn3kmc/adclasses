# use_r_stl.R

# import libraries
library("stlplus")
library("imputeTS")

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)

# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

# convert BeginTime column to time 
mydata$timestamp <- as.POSIXct(mydata$timestamp)

# Convert to numerics
# n.p.
np = as.numeric(myArgs[2])
# s.window
swindow = as.numeric(myArgs[3])
# outer
outer = as.numeric(myArgs[4])
# missing
missing = as.logical(myArgs[5])
# fill_option
fill_option = myArgs[6]

# use stl
stltest <- stlplus(mydata$value, t=mydata$timestamp, n.p=np,s.window=swindow,outer=outer)

stltest$data$timestamp <- mydata$timestamp

if (missing) {
    cat("  missing. will fill. ")
    remainder_filled = na.interpolation(stltest$data$remainder,option=fill_option)
    stltest$data$remainder_filled <- remainder_filled
} else {
    cat(" no missing ")
}

write.csv(file="r_dfs/df_to_csv_stl.csv",x=stltest$data)

cat("True")
