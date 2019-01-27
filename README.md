# adclasses

## How to get adclasses up and running:

1. git clone the directory and cd to directory

2. set up python virtual environment:

> virtualenv -p /usr/bin/python3.5 adclasses_venv

3. activate the virtual environment:

> source adclasses_venv/bin/activate

4. install python dependencies via pip:

> pip install -r requirements.txt

5. install r dependencies (Twitter AnomalyDetection, stlplus):

> install.packages("devtools")

> library(devtools)

> devtools::install_github("twitter/AnomalyDetection")

> devtools::install_github("hafen/stlplus")

> install.packages("imputeTS")


## What anomaly detection methods are considered?

### Univariate

1. Windowed Gaussian

2. SARIMA

    * Guided
        * no seasonality, no trend: ARIMA(0,1,1)
        * no seasonality, trend: ARIMA(0,2,2)
        * seasonality, no trend: SARIMA(0,1,s+1)x(0,1,0,s)
        * seasonality, trend: HWES
   * autoarima (Parameters are determined by running Pyramid only on the time series in the probationary period. If the seasonality parameter s is large, memory issues may occur.)

3. Prophet

4. Multi-step Forecasting RNN

5. Twitter AnomalyDetection (Seasonality must be present.)

6. HOT-SAX

7. Donut (Can deal with missing data points innately.)

8. STL Residual Thresholding (Seasonality must be present. Can deal with missing data points innately.)

9. HTMs

## Notes on anomaly scores:

Every anomaly detection method considered returns an anomaly score between 0 and 1 or is adjusted to return a score (via a Q-function). A sliding window computes the probabilities from a Gaussian distribution with a mean and standard deviation determined from this window. Various window sizes can (and should) be experimented with.

For the windowed Gaussian, the anomaly score is based on the Q-function and is the tail probability of the Gaussian distribution. For SARIMA, Prophet, and RNNs, the prediction error is fed to the Q-function instead. For Donut, the reconstruction probability (which is not well-defined) is used for the Q-function, and for STL, the residuals are used. For Twitter and HOT-SAX, which return a 1 (if a point is an anomaly) or 0 (if not), this output is the anomaly score. For HOT-SAX, we use the first point of the discord. 

A threshold needs to be set to determine anomalies.

## Notes on seasonality:

1. Statsmodels' sarima(x) represents lack of seasonality using s = 0, but Pyramid autoarima represents lack of seasonality using s = 1. In adclasses, we follow statsmodels and represent lack of seasonality using s=0 in sarima(). s=0 is our default setting.

2. R's stlplus requires the seasonality parameter to be AT LEAST 4. This means that techniques that rely on stlplus (like stl_resid_()) can only be used if seasonality is present.

3. Pyramid's autoarima is very slow and memory intensive for large time series with seasonality. Parameters are determined by running Pyramid only on the time series in the probationary period. If seasonality is present and s i large, memory issues can occur.

4. Twitter AD only works on seasonal datasets.