# use_r_autoarima.R
# this is no longer used

library("forecast")

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)

# csv
mydata <- read.csv(file=myArgs[1],stringsAsFactors=FALSE)

seasonality = as.logical(myArgs[2])

tsData = ts(mydata$value, frequency = 24)
fit <- auto.arima(y=tsData,seasonal=seasonality)

# fit <- auto.arima(y=mydata$value,seasonal=seasonality)

# "A compact form of the specification, as a vector giving the number of 
# AR, MA, seasonal AR and seasonal MA coefficients, 
# plus the period and the number of non-seasonal and seasonal differences.
# https://stats.stackexchange.com/questions/178577/how-to-read-p-d-and-q-of-auto-arima
# p, q, P, Q, s, d, D,
# fit$arma

cat(fit$arma)

write.csv(file="r_dfs/df_to_csv_autoarima.csv",x=fit$arma)

cat("True")

# https://stats.stackexchange.com/questions/213201/seasonality-not-taken-account-of-in-auto-arima
# https://stackoverflow.com/questions/46496630/missing-values-arima-model
# https://stats.stackexchange.com/questions/104565/how-to-use-auto-arima-to-impute-missing-values