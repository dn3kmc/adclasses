import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
from sklearn.model_selection import TimeSeriesSplit
import copy
from fbprophet import Prophet
import matplotlib.pyplot as plt
from keras.layers import Dense, GRU
from keras.models import Sequential, load_model
from keras import optimizers
from itertools import zip_longest
import subprocess
import scipy.stats as sct
from saxpy.hotsax import find_discords_hotsax
from scipy.optimize import minimize
import pyramid as pm
from sklearn.metrics import mean_squared_error, mean_squared_log_error, precision_score, recall_score, confusion_matrix
from donut import complete_timestamp, standardize_kpi
import tensorflow as tf
from donut import Donut
from tensorflow import keras
from tfsnippet.modules import Sequential as Sq
from donut import DonutTrainer, DonutPredictor
import math
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm
import joblib
import os


# HAVE TO MAKE CHANGE TO NAB function
# convertAnomalyScoresToDetections in util.py
# create resulting csv to be put into NAB results folder
def create_result_csv(prepath, folder, method_name, dataset_name, date_format, timestep, true_outlier_dates, fill=True):
    no_fill_list = ["ambient_temperature_system_failure",
                    "ec2_cpu_utilization_ac20cd",
                    "exchange-2_cpc_results",
                    "exchange-2_cpm_results",
                    "exchange-3_cpm_results",
                    "ibm-common-stock-closing-prices",
                    "rds_cpu_utilization_cc0c53"]
    if dataset_name in no_fill_list:
        if not fill:
            output_dict_anomaly_scores = joblib.load("anomaly_scores/" + method_name + "/" + dataset_name + "_no_fill")
        else:
            output_dict_anomaly_scores = joblib.load("anomaly_scores/" + method_name + "/" + dataset_name)
    else:
        output_dict_anomaly_scores = joblib.load("anomaly_scores/" + method_name + "/" + dataset_name)

    path = "data/" + dataset_name + ".csv"
    data = pd.read_csv(path, header=0)

    if dataset_name == "ibm-common-stock-closing-prices":
        data["timestamp"] = data["Date"]
        del data["Date"]
        data["value"] = data["IBM common stock closing prices"]
        del data["IBM common stock closing prices"]
    elif dataset_name == "all_data_gift_certificates":
        data["timestamp"] = data["BeginTime"].values
        del data["BeginTime"]
        data["value"] = data["Count"].values
        del data["Count"]
    elif dataset_name == "FARM_Bowling-Green-5-S_Warren":
        data["UTME"] = pd.to_datetime(data["UTME"], format="%Y-%m-%d %H:%M:%S UTC")
        del data["NET"]
        del data["STID"]
        tair_data = pd.DataFrame({"UTME": data["UTME"], "TAIR": data["TAIR"]})
        tair_data.set_index("UTME", inplace=True)
        tair_data = tair_data.interpolate()
        resampled_tair_data = tair_data.resample('30Min', how={'TAIR': np.mean})
        resampled_tair_data["TAIR"] = resampled_tair_data["TAIR"].apply(round_2)
        renamed_resampled_tair_data = pd.DataFrame({"timestamp": resampled_tair_data.index, "value": resampled_tair_data["TAIR"]})
        data = renamed_resampled_tair_data
    else:
        pass

    data["timestamp"] = pd.to_datetime(data["timestamp"], format=date_format)
    start_date = data["timestamp"].values[0]
    end_date = data["timestamp"].values[-1]

    if fill:
        data = fill_missing_time_steps(start_date, end_date, timestep, data, method="linear")
    else:
        data = data.drop_duplicates(subset="timestamp")

    ad = UnivariateAnomalyDetection(dataframe=data, timestep=timestep, dateformat=date_format, name=dataset_name)

    if not fill:
        true_outlier_indices = []
        for date in true_outlier_dates:
            index = data.loc[data["timestamp"] == date].index[0]
            true_outlier_indices.append(index)
        labels = [0] * len(data)
        for index in true_outlier_indices:
            labels[index] = 1

        fake_data = fill_missing_time_steps(start_date, end_date, timestep, data, method="return_nan")
        fake_data["anomaly_score"] = output_dict_anomaly_scores["Anomaly Scores"]
        fake_data = fake_data.dropna()

        result_df = pd.DataFrame({"timestamp": data["timestamp"].values,
                                  "value": data["value"].values,
                                  "anomaly_score": fake_data["anomaly_score"].values,
                                  "label": labels})

    else:
        true_outlier_indices = ad.convert_true_outlier_date(true_outlier_dates)
        labels = [0] * ad.get_length()
        for index in true_outlier_indices:
            labels[index] = 1
        result_df = pd.DataFrame({"timestamp": data["timestamp"],
                                  "value": data["value"],
                                  "anomaly_score": output_dict_anomaly_scores["Anomaly Scores"],
                                  "label": labels})

    if dataset_name in no_fill_list:
        if not fill:
            result_df.to_csv(os.path.join(prepath, method_name, folder, method_name + "_" + dataset_name + "_no_fill.csv"), index=False)
    else:
        result_df.to_csv(os.path.join(prepath, method_name, folder, method_name + "_" + dataset_name + ".csv"), index=False)


def round_2(num):
    return round(num, 2)


# given a dataset that NAB has not seen before,
# helper function to fill out combined_windows.json
# a window is centered around an anomaly!
# the window length is .1 * length of data / number of outliers (not fixed to 2)
def get_window_ends(outlier_date, data_length, num_outliers, format, d):
    """
    Input:

    outlier_date = Outlier date

    data_length = length of the dataset

    num_outliers = number of ground truth outliers

    format = date format

    d = time step size which must be a timedelta type

    Output:

    Returns the start and end dates of the window around the given anomaly
    This should be put into labels/combined_windows.json in the NAB Repository
    """
    middle_time = pd.to_datetime(outlier_date, format=format)
    half_window_length = int((.1 * data_length / num_outliers) / 2)
    # print(half_window_length)
    end = half_window_length * d + middle_time
    start = middle_time - half_window_length * d
    return start, end


def determine_concept_drift(data):
    R1, maxes = oncd.online_changepoint_detection(data["value"], partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
    # choose 95th percentile for vmax
    sparsity = 5  # only plot every fifth data for faster display
    unflattened_post_probs = -np.log(R1[0:-1:sparsity, 0:-1:sparsity])
    post_probs = (-np.log(R1[0:-1:sparsity, 0:-1:sparsity])).flatten()
    chosen_vmax = int(np.percentile(post_probs, 5))
    # plt.axvline(x=data["timestamp"].values[int(len(data) * .15)], color="black", label="probationary line")
    plt.plot(data["timestamp"], data["value"], alpha=.5, color="blue", label="data")
    plt.xticks(rotation=90)
    # plt.legend()
    plt.savefig("dmkd_cd_1.eps", bbox_inches='tight')
    plt.show()
    plt.pcolor(np.array(range(0, len(R1[:, 0]), sparsity)),
               np.array(range(0, len(R1[:, 0]), sparsity)),
               unflattened_post_probs,
               cmap=cm.gray, vmin=0, vmax=chosen_vmax)
    plt.xlabel("time steps")
    plt.ylabel("run length")
    cbar = plt.colorbar(label="P(run)")
    cbar.set_ticks([0, chosen_vmax])
    cbar.set_ticklabels([1, 0])
    # black = highest prob
    # white = lowest prob
    # the colors mean the same as in paper
    # the bar direction is just reversed
    plt.savefig("dmkd_cd_2.eps", bbox_inches='tight')
    plt.show()


def determine_trend(data, maxlag_choice="default"):
    # H0 = trend
    # Ha = no trend
    # p-value < .05. reject null.
    # there is no trend
    print("\n")
    if maxlag_choice == "default":
        maxlag_choice = max(12 * (len(data["value"]) / 100)**(1 / 4), 200)
        # print(len(data))
        # print(type(data))
        if maxlag_choice > len(data):
            maxlag_choice = int(len(data) / 2)
    print("Maxlag:  ", maxlag_choice)
    result = adfuller(data["value"], maxlag=maxlag_choice)
    print('p-value: %f' % result[1])
    if result[1] < .05:
        print("There is no trend!")
    else:
        print("Possible trend.")


def determine_seasonality(data, lags):
    for lag in lags:
        print("----Lag: " + str(lag) + "----")
        plot_acf(data["value"], lags=lag)
        plt.show()


# https://github.com/numenta/NAB/blob/master/nab/detectors/gaussian/windowedGaussian_detector.py
def determine_anomaly_scores_error(actual, predictions, probationary_index, data_length, window_size):
    anomaly_scores = []
    window_data = []
    step_buffer = []
    mean = 0
    std = 1
    step_size = 100
    for i in range(data_length):
        anomaly_score = 0.0
        input_value = actual[i] - predictions[i]
        if len(window_data) > 0:
            anomaly_score = 1 - normal_probability(input_value, mean, std)
        if len(window_data) < window_size:
            window_data.append(input_value)
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std == 0.0:
                std = .000001
        else:
            step_buffer.append(input_value)
            if len(step_buffer) == step_size:
                # slide window forward by step_size
                window_data = window_data[step_size:]
                window_data.extend(step_buffer)
                # reset step_buffer
                step_buffer = []
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std == 0.0:
                    std = .000001
        anomaly_scores.append(anomaly_score)

    return anomaly_scores


def normal_probability(x, mean, std):
    # Given the normal distribution specified by the mean and standard deviation
    # args, return the probability of getting samples > x. This is the
    # Q-function: the tail probability of the normal distribution.
    if x < mean:
        # Gaussian is symmetrical around mean, so flip to get the tail probability
        xp = 2 * mean - x
        return normal_probability(xp, mean, std)
    # Calculate the Q function with the complementary error function, explained
    # here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
    z = (x - mean) / std
    return 0.5 * math.erfc(z / math.sqrt(2))


def precision_recall_curve_info(anomaly_scores_dict, true_outlier_indices_dict):
    """
    Input:

    anomaly_scores_dict = dictionary
    key = name of dataset
    value = list of anomaly scores for the dataset
    Note: if there are missing time steps,
    an anomaly score should be generated for them too
    or you can fill them with 0s.

    true_outlier_indices_dict = dictionary
    key = name of dataset
    value = INDICES of true outliers for that dataset
    Note: true_outlier_indices_dict and anomaly_scores_dict
    should have the same dataset keys

    Output:

    dictionary containing precision and recall of thresholds
    """
    # create true_outlier_dict where
    # key = dataset name
    # value = true outliers binarized
    true_outlier_dict = {}
    for dataset in true_outlier_indices_dict:
        anomaly_scores = anomaly_scores_dict[dataset]
        true_outliers = list(np.zeros(len(anomaly_scores)))
        for outlier_index in true_outlier_indices_dict[dataset]:
            true_outliers[outlier_index] = 1
        true_outlier_dict[dataset] = true_outliers

    # anomaly scores of all datasets in one list
    all_anomaly_scores = []
    for dataset in anomaly_scores_dict:
        anomaly_scores = list(anomaly_scores_dict[dataset])
        all_anomaly_scores += anomaly_scores
    thresholds = []
    for i in range(1, 11):
        thresholds.append(i / 10)

    pr_dict = {}

    for threshold in thresholds:

        for dataset in anomaly_scores_dict:

            anomaly_scores = anomaly_scores_dict[dataset]

            # predicted outliers at this currently
            # considered threshold over this current dataset
            predicted_outliers = []

            for k in range(len(anomaly_scores)):
                # nothing in the probationary period
                # is labeled as an outlier
                if k < int(.15 * len(anomaly_scores)):
                    predicted_outliers.append(0)
                else:
                    score = anomaly_scores[k]
                    if score >= threshold:
                        predicted_outliers.append(1)
                    else:
                        predicted_outliers.append(0)

            # we count the total number of window mistakes
            # for this current dataset weighted
            num_gt_outliers = sum(true_outlier_dict[dataset])
            true_windows = get_window(true_outlier_dict[dataset], num_gt_outliers=num_gt_outliers, window_size="default")
            predicted_windows = get_window(predicted_outliers, num_gt_outliers=num_gt_outliers, window_size="default")
            this_threshold_window_overall_precision, this_threshold_window_overall_recall = get_window_precision_recall(true_windows, predicted_windows)
            pr_dict[threshold] = {"precision": this_threshold_window_overall_precision, "recall": this_threshold_window_overall_recall}

    return pr_dict


# a threshold is chosen for a method, behavior combo
# e.g. an anomaly score threshold is chosen for
# RNN under concept drift with 3 datasets
def thresholding(anomaly_scores_dict, true_outlier_indices_dict, weight_fp=1, weight_fn=1):
    """
    Input:

    anomaly_scores_dict = dictionary
    key = name of dataset
    value = list of anomaly scores for the dataset
    Note: if there are missing time steps,
    an anomaly score should be generated for them too
    or you can fill them with 0s.

    true_outlier_indices_dict = dictionary
    key = name of dataset
    value = INDICES of true outliers for that dataset
    Note: true_outlier_indices_dict and anomaly_scores_dict
    should have the same dataset keys

    weight_fp = weight of a false positive

    weight_fn = weight of a false negative

    Output:

    dictionary containing best results
    """
    # create true_outlier_dict where
    # key = dataset name
    # value = true outliers binarized
    true_outlier_dict = {}
    for dataset in true_outlier_indices_dict:
        anomaly_scores = anomaly_scores_dict[dataset]
        true_outliers = list(np.zeros(len(anomaly_scores)))
        for outlier_index in true_outlier_indices_dict[dataset]:
            true_outliers[outlier_index] = 1
        true_outlier_dict[dataset] = true_outliers

    # anomaly scores of all datasets in one list
    all_anomaly_scores = []
    for dataset in anomaly_scores_dict:
        anomaly_scores = list(anomaly_scores_dict[dataset])
        all_anomaly_scores += anomaly_scores
    minimum = int(round(min(all_anomaly_scores), 4) * 10000)
    thresholds = [i / 10000 for i in range(minimum, 10001, 1)]

    # initialize the best threshold to every label
    # being a mistake
    # note that this is actually the worst case
    # where every point is a window
    best_threshold_window_mistakes_count = len(all_anomaly_scores)

    for threshold in thresholds:

        # if the maximum anomaly scores out of
        # all considered datasets is smaller
        # than the considered threshold
        # stop bc best_threshold_pt_mistakes_count
        # won't change anymore
        if max(all_anomaly_scores) < threshold:
            break

        # at the currently considered threshold
        # we will count the total number of window mistakes
        # across all datasets
        threshold_window_mistakes_count = 0

        # initialize predicted outliers at this currently
        # considered threshold over all datasets
        all_predicted_outliers = {}

        for dataset in anomaly_scores_dict:

            anomaly_scores = anomaly_scores_dict[dataset]

            # predicted outliers at this currently
            # considered threshold over this current dataset
            predicted_outliers = []

            for k in range(len(anomaly_scores)):
                # nothing in the probationary period
                # is labeled as an outlier
                if k < int(.15 * len(anomaly_scores)):
                    predicted_outliers.append(0)
                else:
                    score = anomaly_scores[k]
                    if score >= threshold:
                        predicted_outliers.append(1)
                    else:
                        predicted_outliers.append(0)

            # we count the total number of window mistakes
            # for this current dataset weighted
            num_gt_outliers = sum(true_outlier_dict[dataset])
            true_windows = get_window(true_outlier_dict[dataset], num_gt_outliers=num_gt_outliers, window_size="default")
            predicted_windows = get_window(predicted_outliers, num_gt_outliers=num_gt_outliers, window_size="default")
            tn, fp, fn, tp = confusion_matrix(true_windows, predicted_windows).ravel()
            threshold_window_mistakes_count += (fp * weight_fp + fn * weight_fn)

            # predicted outliers at this currently
            # considered threshold over all datasets
            all_predicted_outliers[dataset] = predicted_outliers

        # determine best window results
        if threshold_window_mistakes_count < best_threshold_window_mistakes_count:
            best_threshold_window_mistakes_count = threshold_window_mistakes_count
            best_threshold_window = threshold

            # x and y combine the lists
            # to get overall precision and recall
            # for all datasets
            x = []
            y = []

            # capture individual dataset stats
            individual_window_precisions = {}
            individual_window_recalls = {}
            individual_window_fps = {}
            individual_window_fns = {}

            for dataset in all_predicted_outliers:

                x += all_predicted_outliers[dataset]
                y += true_outlier_dict[dataset]

                num_gt_outliers = sum(true_outlier_dict[dataset])
                true_windows = get_window(true_outlier_dict[dataset], num_gt_outliers=num_gt_outliers, window_size="default")
                predicted_windows = get_window(all_predicted_outliers[dataset], num_gt_outliers=num_gt_outliers, window_size="default")
                individual_window_precisions[dataset], individual_window_recalls[dataset] = (precision_score(true_windows, predicted_windows), recall_score(true_windows, predicted_windows))
                _, individual_window_fps[dataset], individual_window_fns[dataset], _ = confusion_matrix(true_windows, predicted_windows).ravel()

            best_threshold_window_overall_precision, best_threshold_window_overall_recall = get_window_precision_recall(y, x)

    return {"best_threshold_window_mistakes_count": best_threshold_window_mistakes_count,
            "best_threshold_window": best_threshold_window,
            "best_threshold_window_overall_precision": best_threshold_window_overall_precision,
            "best_threshold_window_overall_recall": best_threshold_window_overall_recall,
            "individual_window_precisions": individual_window_precisions,
            "individual_window_recalls": individual_window_recalls,
            "individual_window_fps": individual_window_fps,
            "individual_window_fns": individual_window_fns}


def pretty_print_results(ad_dict, anomaly_scores_dict, true_outlier_indices_dict, weight_fp=1, weight_fn=1):
    """
    Input:

    ad_dict = dictionary
    key = name of dataset
    value = ad object for that dataset

    anomaly_scores_dict = dictionary
    key = name of dataset
    value = list of anomaly scores for the dataset
    Note: if there are missing time steps,
    an anomaly score should be generated for them too
    or you can fill them with 0s.

    true_outlier_indices_dict = dictionary
    key = name of dataset
    value = INDICES of true outliers for that dataset
    Note: true_outlier_indices_dict and anomaly_scores_dict
    should have the same dataset keys

    weight_fp = weight of a false positive

    weight_fn = weight of a false negative

    Output:

    None, but print and plot lots of results
    """
    best_results = thresholding(anomaly_scores_dict, true_outlier_indices_dict, weight_fp, weight_fn)

    print("\n\n----Window-Based Results----")
    print("Minimum number of mistakes: ", best_results["best_threshold_window_mistakes_count"])
    print("Best window threshold: ", best_results["best_threshold_window"])
    print("Corresponding window overall precision: ", best_results["best_threshold_window_overall_precision"])
    print("Corresponding window overall recall: ", best_results["best_threshold_window_overall_recall"])
    print("Corresponding window overall F score: ", get_f_score(best_results["best_threshold_window_overall_precision"], best_results["best_threshold_window_overall_recall"]))

    for dataset in anomaly_scores_dict:
        print("\n\n")
        print("---", dataset, "---")

        anomaly_scores = anomaly_scores_dict[dataset]
        true_outlier_indices = true_outlier_indices_dict[dataset]
        ad = ad_dict[dataset]

        print("Corresponding window precision: ", best_results["individual_window_precisions"][dataset])
        print("Corresponding window recall: ", best_results["individual_window_recalls"][dataset])
        print("Corresponding window false positives: ", best_results["individual_window_fps"][dataset])
        print("Corresponding window false negatives: ", best_results["individual_window_fns"][dataset])

        predicted_outlier_indices = determine_outliers(best_results["best_threshold_window"], anomaly_scores)
        predicted_outlier_indices = remove_probationary_outliers(len(anomaly_scores), predicted_outlier_indices)
        print("\n\nOutlier indices as determined by best window threshold: ", predicted_outlier_indices)
        ad.plot_outliers(predicted_outlier_indices, true_outlier_indices)

    return {"Minimum number of mistakes": best_results["best_threshold_window_mistakes_count"],
            "Best window threshold": best_results["best_threshold_window"],
            "Corresponding window overall precision": best_results["best_threshold_window_overall_precision"],
            "Corresponding window overall recall": best_results["best_threshold_window_overall_recall"],
            "Corresponding window overall F score": get_f_score(best_results["best_threshold_window_overall_precision"], best_results["best_threshold_window_overall_recall"])}


def pretty_print_end_results(end_results):
    for item in end_results:
        print("False Positive Weight: ", item)
        for item1 in end_results[item]:
            print("\t", item1, ":", end_results[item][item1])


def determine_outliers(threshold, anomaly_scores):
    """Return outlier indices given a threshold on the anomaly scores"""
    outlier_indices = []
    for i in range(len(anomaly_scores)):
        score = anomaly_scores[i]
        if score >= threshold:
            outlier_indices.append(i)
    return outlier_indices


def rmse(predictions, targets):
    """Evaluate root mean square error"""
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_window(a_list, num_gt_outliers=2, window_size="default"):
    """
    Input:

    a_list
    e.g. [0,0,1,0,1...]

    window size for anomaly detection
    a WINDOW is now anomalous if a point in it is anomalous

    num_gt_outliers = number of ground truth outliers
    necessary to determine default window size

    Output:

    a_list windowed
    e.g. if window_size = 2 and a_list = [0,0,0,1,1,0,0,...]
    we would return [0,1,1]
    """
    if num_gt_outliers == 0:
            raise ValueError("Cannot calculate window precision and recall if no outliers.")
    # determine window size
    if window_size == "default":
        window_size = int((.1 * len(a_list)) / num_gt_outliers)
    else:
        if window_size > len(a_list):
            raise ValueError("Given window size is too large.")
    # print("Window size", window_size)
    number_of_windows = int(len(a_list) / window_size)
    chunked_list = np.array_split(a_list, number_of_windows)
    windowed_list = []
    for chunk in chunked_list:
        if 1 in chunk:
            windowed_list.append(1)
        else:
            windowed_list.append(0)
    return windowed_list


def get_window_precision_recall(true_outliers, predicted_outliers, window_size="default"):
    """
    Input:

    true_outliers: if there is a 1 at index i that means there was an outlier at time step i
    e.g. [0,0,1,0,1...]

    predicted_outliers: if there is a 1 at index i that means there we predicted there was an outlier at time step i
    e.g. [0,0,1,1,1...]

    window size for anomaly detection
    a WINDOW is now anomalous if a point in it is anomalous

    Output:

    (precision, recall) window wise
    """
    num_gt_outliers = sum(true_outliers)
    true_windows = get_window(true_outliers, num_gt_outliers=num_gt_outliers, window_size="default")
    predicted_windows = get_window(predicted_outliers, num_gt_outliers=num_gt_outliers, window_size="default")
    return (precision_score(true_windows, predicted_windows), recall_score(true_windows, predicted_windows))


def get_f_score(precision, recall):
    """Get F score from precision and recall"""
    return 2 * ((precision * recall) / (precision + recall))


def convert_outlier_index(df_length, outlier_index_list_1s):
    """
    Input:

    df_length = length of dataframe

    list of predicted outlier indices
    e.g. [2,5]

    Output:

    list of whether or not an outlier
    e.g. [0,0,1,0,0,1,...] for the above example
    """
    outlier_index_list = np.zeros(df_length)
    for outlier_index in outlier_index_list_1s:
        outlier_index_list[outlier_index] = 1
    return outlier_index_list


def remove_probationary_outliers(df_length, predicted_outlier_indices):
    """Remove all predicted outliers in probationary zone"""
    cleaned_outliers = []
    for outlier_index in predicted_outlier_indices:
        if outlier_index >= int(.15 * df_length):
            cleaned_outliers.append(outlier_index)
    return cleaned_outliers


def use_r_netflixsurus(dataframe, dimension_name, freq, autodiff, forcediff, scale, lpenalty, spenalty):
    """
    Helper function to determine netflix surus outliers.

    Uses RADS, a R library.
    """
    # if there is seasonality
    if freq != 1:
        # if the length of the time series is NOT a multiple of
        # the seasonality parameter
        # we truncate FROM THE FRONT
        if len(dataframe) % freq != 0:
            truncate = True
            truncated_dataframe = dataframe[len(dataframe) % freq:]
            # write dataframe to a csv
            simple_dataframe = pd.DataFrame({"timestamp": truncated_dataframe["timestamp"],
                                             "value": truncated_dataframe[dimension_name]})
        else:
            truncate = False
            # write dataframe to a csv
            simple_dataframe = pd.DataFrame({"timestamp": dataframe["timestamp"],
                                             "value": dataframe[dimension_name]})

    else:
        truncate = False
        # write dataframe to a csv
        simple_dataframe = pd.DataFrame({"timestamp": dataframe["timestamp"],
                                         "value": dataframe[dimension_name]})

    simple_dataframe.to_csv("r_dfs/df_to_csv_netflixsurus.csv", header=True)

    # arguments needed to give to R:
    # name of csv, frequency, autodiff, forcediff, scale, lpenalty, spenalty

    if autodiff:
        autodiff_str = "TRUE"
    else:
        autodiff_str = "FALSE"

    if forcediff:
        forcediff_str = "TRUE"
    else:
        forcediff_str = "FALSE"

    args = ['r_dfs/df_to_csv_netflixsurus.csv',
            str(freq),
            autodiff_str,
            forcediff_str,
            str(scale),
            str(lpenalty),
            str(spenalty)]

    command = 'Rscript'
    path2script = 'use_r_netflixsurus.R'
    # Build subprocess command
    cmd = [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)
    print('R netflix surus completed:', x)
    outlier_indices_1_indexed = pd.read_csv("r_dfs/netflix_surus_outliers_index_1.csv", header=0)
    # warning! R is 1 indexed not 0 indexed!
    # so if R thinks t is an outlier, it is actually t-1 in Python
    indices = []
    for index in outlier_indices_1_indexed.values:
        # if truncation is true, the outlier indices are off
        if truncate:
            indices.append(index[1] - 1 + len(dataframe) % freq)
        else:
            indices.append(index[1] - 1)
    return indices


def use_r_twitterad(dataframe, dimension_name, max_anoms,
                    direction, alpha, period):
    """
    Helper function to determine Twitter AD outlier indices.

    Twitter AD is a R library. We use vec instead of ts because ts makes dumb
    assumptions on what kind of time steps can be used. More specifically,
    ts can only use 1 second, 1 minute, 1 hour, or 1 day time steps. So something
    like 30 minute time steps will fail. This is why we use vec instead. vec also
    requires a period parameter compared to ts. This period parameter has to be greater
    than 1. This is dumb bc it means we must have seasonal datasets
    to use Twitter AD. Dumb.
    """
    # write dataframe to a csv
    simple_dataframe = pd.DataFrame({"timestamp": dataframe["timestamp"],
                                     "value": dataframe[dimension_name]})

    simple_dataframe.to_csv("r_dfs/df_to_csv_twitter.csv", header=True)

    # arguments needed to give to R:
    # name of csv, max_anoms,
    # direction, alpha, period

    args = ['r_dfs/df_to_csv_twitter.csv',
            str(max_anoms),
            direction,
            str(alpha),
            str(period)]
    command = 'Rscript'
    path2script = 'use_r_twitterad.R'
    # Build subprocess command
    cmd = [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)
    print('R twitterad completed:', x)
    outlier_indices = pd.read_csv("r_dfs/twitter_index_anomalies.csv", header=0)
    return outlier_indices


def make_context_variables_x(context_variable_names, data):
    """
    Helper function for Liu's outlier method

    Creates contextual variable vector
    """
    x = np.empty((1, len(context_variable_names)))
    for i in range(len(data)):
        ctv = []
        for name in context_variable_names:
            ctv.append(int(data[name].values[i]))
        x = np.vstack([x, ctv])
    # word of warning: the first element is full of random things
    # i.e. np.empty is not really empty
    x = x[1:]
    return x


def step_two(context_variable_names, z, x):
    """Determine anomaly scores for Liu method"""
    # initialize
    # st is the identity matrix so
    # st_inverse is also the identity matrix
    old_st_inverse = np.eye(len(context_variable_names))
    old_mt = np.array(len(context_variable_names) * [0])
    old_at = 1
    old_bt = 100

    v = []
    theta_squareds = []
    mus = []

    # iterate
    for time_step in range(len(z) - 1):
        # print(time_step)

        # if we we did x[time_step + 1] * x[time_step + 1] it would be element wise multiplication
        # np.outer makes it so it is actually matrix multiplication
        new_st_inverse = old_st_inverse + np.outer(x[time_step + 1], x[time_step + 1])

        new_st = np.linalg.inv(new_st_inverse)
        # note that np.dot of two matrices means matrix multiplication
        # note that z[time_step + 1] * x[time_step + 1] is ok bc the first is a scalar
        new_mt = new_st.dot(old_st_inverse.dot(old_mt) + z[time_step + 1] * x[time_step + 1])

        new_at = old_at + .5

        new_bt = old_bt + .5 * (z[time_step + 1]**2 -
                                new_mt.dot(new_st_inverse.dot(new_mt)) +
                                old_mt.dot(old_st_inverse.dot(old_mt)))

        # refresh
        old_st_inverse = new_st_inverse
        old_mt = new_mt
        old_at = new_at
        old_bt = new_bt

        nu = 2 * old_at
        mu = x[time_step + 1].dot(old_mt)
        mus.append(mu)

        old_st = np.linalg.inv(old_st_inverse)
        theta_squared = (old_bt / old_at) * (1 + x[time_step + 1].dot(old_st.dot(x[time_step + 1])))
        theta_squareds.append(theta_squared)
        boundary = abs(z[time_step + 1] - mu) / np.sqrt(theta_squared)

        # outlier score for z[time_step + 1]
        # 1-ICDF
        # https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p
        outlier_score = 1 - sct.t.sf(x=boundary, df=nu, loc=mu, scale=theta_squared)
        v.append(outlier_score)

    return v, mus, theta_squareds


def use_r_stl(dataframe, dimension_name, n_periods, s_window, outer, missing, fill_option):
    """
    STL in Python sucks. Use R's stlplus instead.

    Input:

    dataframe (must have timestamp column)

    dimension_name = name of field to do stlplus to

    n_periods = periodicity of seasonality

    s_window = smoothing parameter for seasonal filter

    outer = number of cycles thru outer loop, more cycles reduces affect of outliers

    missing = there are NaNs and we will fill them out in the stl remainder

    fill_option = how to fill NaNs in residuals

    Output:

    STL remainder of data's main variable
    """
    # write dataframe to a csv
    simple_dataframe = pd.DataFrame({"timestamp": dataframe["timestamp"],
                                     "value": dataframe[dimension_name]})

    simple_dataframe.to_csv("r_dfs/df_to_csv.csv", header=True)

    # arguments needed to give to R:
    # name of csv, n_periods, s.window, outer, missing, fill_option
    if missing:
        miss = "TRUE"
    else:
        miss = "FALSE"
    args = ['r_dfs/df_to_csv.csv', str(n_periods), str(s_window), str(outer), miss, fill_option]

    command = 'Rscript'
    path2script = 'use_r_stl.R'
    # Build subprocess command
    cmd = [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)
    print('R stlplus completed:', x)
    stl_data = pd.read_csv("r_dfs/df_to_csv_stl.csv", header=0)
    if missing:
        return stl_data["raw"].values, stl_data["remainder_filled"].values
    else:
        return stl_data["raw"].values, stl_data["remainder"].values


def grouper(n, iterable, fillvalue=None):
    """
    Example:

    grouper(2,[1,2,3,4]) = [[1,2],[3,4]]

    Helper function for determining anomaly scores for RNNs

    """
    args = [iter(iterable)] * n
    raw = zip_longest(fillvalue=fillvalue, *args)
    done = []
    for item in raw:
        done.append(list(item))
    return done


def det_anomaly_score(actual, predicted, dimension):
    """
    Helper function for determining anomaly scores for RNNs

    See saurav2018online papers

    """
    actual = grouper(dimension, actual[0])
    predicted = grouper(dimension, predicted[0])
    n_seq = len(actual)
    the_sum = 0
    for item in list(zip(actual, predicted)):
        diff = [a_i - b_i for a_i, b_i in zip(item[0], item[1])]
        the_sum += np.linalg.norm(diff, 2)
    return((1 / n_seq) * the_sum)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Helper function for putting data in proper format for RNN

    See https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n_out)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def normalize_data(series, value_name):
    """Normalize the data before using RNN or Liu"""
    raw_values = series[value_name]
    raw_values_mean = np.mean(raw_values)
    raw_values_std = np.std(raw_values)
    scaled_values = []
    for item in raw_values:
        scaled_values.append((item - raw_values_mean) / raw_values_std)
    return scaled_values


def prepare_data(scaled_values, n_test, n_lag, n_seq, to_batch):
    """
    Helper function for putting data in proper format for RNN

    See https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

    """
    # a row in supervised is:
    # var1(t-n_lag), var2(t-n_lag), var3(t-n_lag)...var1(t+n_test-1),var2(t+n_test-1),var3(t+n_test-1)
    # number of columns = (n_lag + n_seq) * dimension of data
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    # each row is its own list in supervised_values
    supervised_values = supervised.values
    if to_batch:
        return supervised_values
    else:
        # the last n_test rows of supervised values is test
        # everything else is train
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return train, test


def fit_rnn(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons, dimensions):
    """Train a multi-step RNN"""
    # reshape training into [samples, timesteps, features]
    x, y = train[:, 0:n_lag * dimensions], train[:, n_lag * dimensions:]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    # design network
    model = Sequential()
    # L = 1,2,3 dropout is .2 and learning rate is .0005 and number of cells in each layer is 20,40
    model.add(GRU(n_neurons, batch_input_shape=(n_batch, x.shape[1], x.shape[2]), dropout=0.2, stateful=True, return_sequences=True))
    model.add(GRU(n_neurons, batch_input_shape=(n_batch, x.shape[1], x.shape[2]), dropout=0.2, stateful=True, return_sequences=True))
    model.add(GRU(n_neurons, batch_input_shape=(n_batch, x.shape[1], x.shape[2]), dropout=0.2, stateful=True))
    model.add(Dense(y.shape[1]))
    # sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.0005, beta_1=.5)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # fit network
    for i in range(nb_epoch):
        # print("Epoch:", i)
        model.fit(x, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    # save model
    return model


def timeseries_cv_score(params, series, loss_function=mean_squared_error, slen=24):
    """
    Determine optimal parameters for HWES

    Input:

    params = vector of parameters for optimization

    series = dataset with timeseries

    slen = season length for Holt-Winters model

    Output:

    Return error on CV
    """
    errors = []
    # values = series.values
    values = series
    alpha, beta, gamma = params
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=slen,
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
    return np.mean(np.array(errors))


def find_missing_time_steps(start_date, end_date, time_step_size, given_dataframe):
    """Determine if there are missing time steps in time series"""
    ref_date_range = pd.date_range(start_date, end_date, freq=time_step_size)
    gaps = ref_date_range[~ref_date_range.isin(given_dataframe["timestamp"])]
    return gaps


def remove_duplicate_time_steps(start_date, end_date, time_step_size, given_dataframe, value="value"):
    """Remove duplicate time steps"""
    given_dataframe_copy = copy.deepcopy(given_dataframe)
    given_dataframe_copy.set_index('timestamp', inplace=True)
    # if duplicate records are found, keep only the first occurrence
    if len((given_dataframe_copy[given_dataframe_copy.index.duplicated()])) != 0:
        print("Duplicate records found: ")
        indexing = given_dataframe_copy[given_dataframe_copy.index.duplicated()].index[0]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
        print("Removing duplicates and keeping:  ")
        given_dataframe_copy = given_dataframe_copy[~given_dataframe_copy.index.duplicated()]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
    df = pd.DataFrame({"timestamp": given_dataframe_copy.index, value: given_dataframe_copy.value})
    return df


def fill_missing_time_steps(start_date, end_date, time_step_size, given_dataframe, value="value", method="linear", order=4):
    """Fill in missing time steps in time series using linear interpolation"""
    ref_date_range = pd.date_range(start_date, end_date, freq=time_step_size)
    # make a copy of the given data frame that
    # will have spaces for the missing dates which we fill with NaNs
    given_dataframe_copy = copy.deepcopy(given_dataframe)
    given_dataframe_copy.set_index('timestamp', inplace=True)
    # if duplicate records are found, keep only the first occurrence
    if len((given_dataframe_copy[given_dataframe_copy.index.duplicated()])) != 0:
        print("Duplicate records found: ")
        indexing = given_dataframe_copy[given_dataframe_copy.index.duplicated()].index[0]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
        print("Removing duplicates and keeping:  ")
        given_dataframe_copy = given_dataframe_copy[~given_dataframe_copy.index.duplicated()]
        print(given_dataframe_copy.loc[given_dataframe_copy.index == indexing])
    given_dataframe_copy = given_dataframe_copy.reindex(ref_date_range, fill_value=np.nan)
    # missing timesteps are added as rows with nan values
    if method == "return_nan":
        df = pd.DataFrame({"timestamp": given_dataframe_copy.index, value: given_dataframe_copy[value].values})
        return df
    # convert to a time series so I can use interpolate
    else:
        test_series = pd.Series(given_dataframe_copy[value].values, index=given_dataframe_copy.index)
        # different interpolation options:
        # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
        # method =
        # 'linear', 'nearest’, ‘zero’, ‘slinear’,
        # ‘quadratic’, ‘cubic’, ‘barycentric’,
        # ‘polynomial’
        #  Both ‘polynomial’ and ‘spline’
        # require that you also specify an order (int),
        if (method == "polynomial") or (method == "spline"):
            test_series_2 = test_series.interpolate(method=method, order=order)
        else:
            test_series_2 = test_series.interpolate(method=method)
        # convert back to a dataframe
        df = pd.DataFrame({"timestamp": test_series_2.index, value: test_series_2.values})
        return df


def det_req_obs(p, d, q, big_p, big_d, big_q, s):
    """Determine if there are enough observations given SARIMA params"""
    return d + big_d * s + max(3 * q + 1, 3 * big_q * s + 1, p, big_p * s) + 1


class HoltWinters:
    """Holt-Winters model class: # see https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3"""

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBound = []
        self.LowerBound = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                self.PredictedDeviation.append(0)
                self.UpperBound.append(self.result[0] + self.scaling_factor * self.PredictedDeviation[0])
                self.LowerBound.append(self.result[0] - self.scaling_factor * self.PredictedDeviation[0])
                continue
            # predicting
            if i >= len(self.series):
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBound.append(self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1])

            self.LowerBound.append(self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])


class UnivariateAnomalyDetection:

    def __init__(self, dataframe, timestep, dateformat, name, missing=False):

        if ("value" not in dataframe) or ("timestamp" not in dataframe):
            raise ValueError("The given dataframe must have 'value' and 'timestamp' columns.")

        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format=dateformat)
        self.dataframe = dataframe
        self.timestep = timestep
        self.dateformat = dateformat
        self.missing = missing
        self.name = name
        self.main_variable_name = "value"

        # If there are missing time steps,
        # make sure that means the row disappears
        # I.E. there should be no nans
        # This is different from ContextualAnomalyDetection class
        # where every row MUST be present,
        # contextual variables cannot have NaNs,
        # but main variables are allowed to have NaNs
        if self.dataframe[self.main_variable_name].isnull().values.any():
            raise ValueError("NaNs present in data. Drop them first.")
        if len(find_missing_time_steps(dataframe["timestamp"].values[0], dataframe["timestamp"].values[-1], timestep, dataframe)) > 0:
            print("WARNING: There are missing time steps. This restricts the kinds of anomaly detection methods that can be used. See fill_missing_time_steps")
            self.missing = True
            # create ANOTHER dataframe where missing timesteps
            # are added as rows and have values of NaN
            # this is necessary for R stlplus to work
            # the reason why we create an entirely new df is bc
            # donut requires the rows to not exist entirely
            self.dataframe_missing = fill_missing_time_steps(self.dataframe["timestamp"].values[0], self.dataframe["timestamp"].values[-1], self.timestep, self.dataframe, method="return_nan")

    def get_length(self):
        """Get length of dataframe"""
        if self.missing:
            return len(self.dataframe_missing)
        else:
            return len(self.dataframe)

    def get_probationary_index(self):
        """Get the index of the first time step after the probationary period"""
        return int(.15 * self.get_length())

    def get_mean(self):
        """Get mean of dataframe"""
        return np.mean(self.dataframe[self.main_variable_name])

    def get_variance(self):
        """Get variance of dataframe"""
        return np.var(self.dataframe[self.main_variable_name])

    def get_std(self):
        """Get standard deviation of dataframe"""
        return np.std(self.dataframe[self.main_variable_name])

    def get_timestep(self):
        """Get time step size of dataframe"""
        return self.timestep

    def get_dateformat(self):
        """Get the string date format of dataframe"""
        return self.dateformat

    def convert_true_outlier_date(self, outlier_date_list):
        """
        Input:

        List of true outlier dates
        e.g. ["2011-07-26 06:00:01"]

        Output:
        Indices of those true outlier dates

        Also makes an outlier column for the dataframe
        """
        if self.missing:
            outlier_index_list = np.zeros(len(self.dataframe_missing))
            only_1s = []
            for date in outlier_date_list:
                outlier_index = self.dataframe_missing.loc[self.dataframe_missing['timestamp'] == date].index[0]
                outlier_index_list[outlier_index] = 1
                only_1s.append(outlier_index)
            self.dataframe_missing["outlier"] = outlier_index_list
            return only_1s
        else:
            outlier_index_list = np.zeros(len(self.dataframe))
            only_1s = []
            for date in outlier_date_list:
                outlier_index = self.dataframe.loc[self.dataframe['timestamp'] == date].index[0]
                outlier_index_list[outlier_index] = 1
                only_1s.append(outlier_index)
            self.dataframe["outlier"] = outlier_index_list
            return only_1s

    def plot_outliers(self, predicted_outlier_index_values, true_outlier_index_values):
        # data
        if self.missing:
            miss_drop_na = self.dataframe_missing.dropna()
            # plt.plot(miss_drop_na.index.values, miss_drop_na[self.main_variable_name].values, color="blue", alpha=.5)
            plt.scatter(miss_drop_na.index.values, miss_drop_na[self.main_variable_name].values, color="blue", alpha=1, s=.8)
        else:
            plt.plot(self.dataframe.index.values, self.dataframe[self.main_variable_name].values, color="blue", alpha=.5)

        # true outliers
        outlier_y_values = []
        for outlier_index in true_outlier_index_values:
            if self.missing:
                outlier_y_values.append(self.dataframe_missing[self.main_variable_name].values[outlier_index])
            else:
                outlier_y_values.append(self.dataframe[self.main_variable_name].values[outlier_index])
        plt.scatter(true_outlier_index_values, outlier_y_values, color="red", label="true outliers", marker="X", alpha=.5, s=100, zorder=100)

        # predicted outliers
        if predicted_outlier_index_values:
            outlier_y_values = []
            for outlier_index in predicted_outlier_index_values:
                if self.missing:
                    outlier_y_values.append(self.dataframe_missing[self.main_variable_name].values[outlier_index])
                else:
                    outlier_y_values.append(self.dataframe[self.main_variable_name].values[outlier_index])
            plt.scatter(predicted_outlier_index_values, outlier_y_values, color="green", label="predicted outliers", alpha=1)
        plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
        # plt.legend()
        plt.savefig("dmkd_plot.eps", bbox_inches='tight')
        plt.show()

    def windowed_gaussian(self, gaussian_window_size, plot_anomaly_score=False):
        """
        Input:

        plot_anomaly_score = True or False.
        Whether or not to plot the anomaly score

        gaussian_window_size = window size for scoring

        Output:

        dictionary:
        anomaly scores, time

        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        anomaly_scores = determine_anomaly_scores_error(self.dataframe["value"], [0] * self.get_length(), self.get_probationary_index(), self.get_length(), gaussian_window_size)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        end = time.time()
        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    # You can either provide p,d,q,P,D,Q,s yourself or
    # use autoarima to have them determined for you,
    # By default, s is 0 (no seasonality).
    # If there is seasonality, regardless of autoarima or not,
    # you should set s appropriately by using an ACF plot.
    # Warning: Statsmodels' SARIMAX represents lack of
    # seasonality using s = 0, but autoarima represents
    # lack of seasonality using s = 1.
    def sarima(self, gaussian_window_size, p=1, d=1, q=1, big_p=0, big_d=0,
               big_q=0, s=0, autoarima=False,
               autoarima_season=True, maxiter=50,
               plot_acf=False, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        SARIMA parameters (p,d,q)x(P,D,Q,s)

        autoarima = True or False.
        Whether or not to use auto.arima
        to determine parameters. If True, the given
        parameters with the exception of s are ignored
        and determined using autoarima

        autoarima_season = True or False.
        Whether or not to allow seasonal behavior
        in autoarima. If True, allows for P,D,Q,s

        maxiter = maximum number of iterations

        plot_acf = True or False.
        Whether or not to plot acf and pacf of data
        and residuals

        plot_anomaly_score = True or False.
        Whether or not to plot the anomaly score

        Output:

        dictionary:
        anomaly scores, time, RMSE

        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()

        # use autoarima to determine p,d,q,etc
        # only use autoarima on the probationary period
        if autoarima:
            # seasonality allowed
            # determine p,d,q,P,D,Q,s
            if autoarima_season:
                arima = pm.auto_arima(self.dataframe[self.main_variable_name].values[0:self.get_probationary_index()], error_action='ignore', trace=1, seasonal=True, m=s, maxiter=maxiter)
                p, d, q = arima.order
                big_p, big_d, big_q, s = arima.seasonal_order
            # seasonality not allowed
            # determine only p,d,q
            else:
                arima = pm.auto_arima(self.dataframe[self.main_variable_name].values[0:self.get_probationary_index()], error_action='ignore', trace=1, maxiter=maxiter)
                p, d, q = arima.order

        if len(self.dataframe) < det_req_obs(p, d, q, big_p, big_d, big_q, s):
            raise ValueError("Not enough observations for given parameters.")

        print("Final parameters: ", (p, d, q, big_p, big_d, big_q, s))

        sarimax_model = sm.tsa.SARIMAX(self.dataframe[self.main_variable_name], order=(p, d, q), seasonal_order=(big_p, big_d, big_q, s))
        sarimax_model_results = sarimax_model.fit(disp=0)
        sarimax_model_forecast = sarimax_model_results.get_prediction()
        predictions = sarimax_model_forecast.predicted_mean
        actual = self.dataframe[self.main_variable_name].values

        anomaly_scores = determine_anomaly_scores_error(actual, predictions, self.get_probationary_index(), self.get_length(), gaussian_window_size)

        end = time.time()

        if plot_acf:
            sm.graphics.tsa.plot_acf(self.dataframe[self.main_variable_name], lags=100)
            plt.title("Data ACF")
            plt.show()

            sm.graphics.tsa.plot_pacf(self.dataframe[self.main_variable_name], lags=100)
            plt.title("Data PACF")
            plt.show()

            sm.graphics.tsa.plot_acf(sarimax_model_results.resid, lags=100)
            plt.title("Residual ACF after fitting to SARIMA")
            plt.show()

            sm.graphics.tsa.plot_pacf(sarimax_model_results.resid, lags=100)
            plt.title("Residual PACF after fitting to SARIMA")
            plt.show()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start,
                "RMSE": rmse(predictions, actual)}

    # see https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3
    def hwes(self, gaussian_window_size, slen, scaling_factor=3, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        slen = length of a season

        scaling_factor = sets the width of the confidence interval by Brutlag
        (usually takes values from 2 to 3)

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        Output:

        dictionary:
        anomaly scores, time, RMSE

        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        x = [0, 0, 0]
        opt = minimize(timeseries_cv_score, x0=x, args=(self.dataframe[self.main_variable_name].values[0:self.get_probationary_index()], mean_squared_log_error), method="TNC", bounds=((0, 1), (0, 1), (0, 1)))
        # alpha, beta, gamma = Holt-Winters model coefficients
        # determined off of optimization from probationary period
        alpha, beta, gamma = opt.x
        # n_preds = how many steps past what we have
        hwes_model = HoltWinters(self.dataframe[self.main_variable_name], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_preds=0, scaling_factor=scaling_factor)
        hwes_model.triple_exponential_smoothing()
        predictions = hwes_model.result
        actual = self.dataframe[self.main_variable_name].values

        anomaly_scores = determine_anomaly_scores_error(actual, predictions, self.get_probationary_index(), self.get_length(), gaussian_window_size)

        end = time.time()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start,
                "RMSE": rmse(predictions, actual)}

    # see https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
    # see https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3
    def facebook_prophet(self, gaussian_window_size, periods, changepoint_prior_scale=.05,
                         growth='linear', yearly_seasonality='auto',
                         weekly_seasonality='auto', daily_seasonality='auto',
                         holidays=None, seasonality_mode='additive',
                         seasonality_prior_scale=10, holidays_prior_scale=10,
                         interval_width=.95, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        periods = how many steps past the given dataframe you want to predict

        changepoint_prior_scale = Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.

        growth = String 'linear' or 'logistic' to specify a linear or logistic
        trend.

        yearly_seasonality = Fit yearly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.

        weekly_seasonality = Fit weekly seasonality.Can be 'auto', True, False, or a number of Fourier terms to generate.

        daily_seasonality = Fit daily seasonality.Can be 'auto', True, False, or a number of Fourier terms to generate.

        holidays = pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.

        seasonality_mode = 'additive' (default) or 'multiplicative'.

        seasonality_prior_scale = Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.

        holidays_prior_scale = Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.

        interval_width = Float, width of the uncertainty intervals provided
        for the forecast.

        plot_anomaly_score =  True or False. Whether or not to plot data with anomaly scores

        Output:

        dictionary:
        anomaly scores, time, RMSE

        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        fb_prophet_model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                                   growth=growth,
                                   yearly_seasonality=yearly_seasonality,
                                   weekly_seasonality=weekly_seasonality,
                                   daily_seasonality=daily_seasonality,
                                   holidays=holidays,
                                   seasonality_mode=seasonality_mode,
                                   seasonality_prior_scale=seasonality_prior_scale,
                                   holidays_prior_scale=holidays_prior_scale,
                                   interval_width=interval_width)
        fb_df = pd.DataFrame({"ds": self.dataframe["timestamp"], "y": self.dataframe[self.main_variable_name]})
        fb_prophet_model.fit(fb_df, verbose=False)
        future = fb_prophet_model.make_future_dataframe(periods=periods, freq=self.timestep)
        fcst = fb_prophet_model.predict(future)
        predictions = fcst["yhat"].values
        actual = self.dataframe[self.main_variable_name].values

        anomaly_scores = determine_anomaly_scores_error(actual, predictions, self.get_probationary_index(), self.get_length(), gaussian_window_size)

        end = time.time()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start,
                "RMSE": rmse(predictions, actual)}

    def rnn(self, gaussian_window_size, n_lag, n_seq, n_test, n_epochs, n_batch, n_neurons, training_index, plot_anomaly_score=False, load=False, save_final=True):
        """
        Input:

        gaussian_window_size = window size for scoring

        n_lag = number of observations as input

        n_seq = number of observations as output

        n_test = the last n_test time steps form the test set

        n_epochs = number of epochs

        n_batch = batch size

        n_neurons = number of cells in each layer

        training_index = to separate training set from rest of set for determining anomaly scores

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        load = True or False. Whether or not to load a pretrained model

        save_final = True or False. Whether or not to save the final model (even after batch training)

        Output:

        dictionary:
        anomaly scores, time
        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")

        if training_index + n_lag + n_seq > len(self.dataframe):
            raise ValueError("training_index + n_lag + n_seq must be less than the length of the dataframe to produce anomaly scores.")

        start = time.time()
        indexed_date_count_df = normalize_data(self.dataframe, self.main_variable_name)
        my_data = indexed_date_count_df[0:training_index]

        # number of columns for train and test is
        # (n_lag + n_seq) * dimensions = n_lag + n_seq
        # since dimensions is 1

        # number of rows for train is
        # training_index-n_lag-n_test-n_seq+1
        # number of rows for test is n_test

        train, test = prepare_data(my_data, n_test, n_lag, n_seq, False)

        if not load:
            model = fit_rnn(train, n_lag, n_seq, n_batch, n_epochs, n_neurons, 1)
            # Saving/loading whole models (architecture + weights + optimizer state)
            model.save("save_rnn/" + self.name + "_initial_rnn.h5")
        else:
            print("WARNING: printed time does not include initial training time")
            model = load_model("save_rnn/" + self.name + "_initial_rnn.h5")

        # anomaly score ARE generated for the probationary period
        # by predicting ON TRAINING DATA
        # if we don't do this then the whole probationary period
        # will have an anomaly score of 0
        # but it will mess with the rolling mean we use to
        # normalize the anomaly scores and make the
        # first few points after the probationary period look
        # super anomalous

        start_point = 0
        error_anomaly_scores_x_values = []
        error_anomaly_scores_y_values = []
        while start_point + n_lag + n_seq < len(indexed_date_count_df):
            end_point = start_point + n_lag + n_seq
            batch_data = indexed_date_count_df[start_point:end_point]
            train_batch = prepare_data(batch_data, n_test, n_lag, n_seq, True)
            x, y = train_batch[:, 0:n_lag], train_batch[:, n_lag:]

            # x goes from (1,100) to (1,1,100)  after reshape
            x = x.reshape(x.shape[0], 1, x.shape[1])
            model.train_on_batch(x, y)
            predictions = model.predict_on_batch(x)

            # x has length n_lag
            # predictions and y are vectors of length n_seq
            # because predictions is a VECTOR
            # we cannot have a pointwise RMSE

            # univariate so dimension is 1
            anomaly_score = det_anomaly_score(y, predictions, 1)
            error_anomaly_scores_x_values.append(end_point)
            error_anomaly_scores_y_values.append(anomaly_score)
            start_point += 1

        padded_error_anomaly_scores_y_values = [0] * (self.get_length() - len(error_anomaly_scores_y_values)) + error_anomaly_scores_y_values

        # save these unnormalized scores! then for various gaussian window sizes
        # you can just use those scores
        save_path = "rnn_unnormalized_anomaly_scores/" + self.name + "_rnn_unnormalized_scores"
        end = time.time()
        joblib.dump({"Unnormalized Anomaly Scores": padded_error_anomaly_scores_y_values, "Time": end - start}, save_path)

        # these scores are normalized using the gaussian window size
        anomaly_scores = determine_anomaly_scores_error(padded_error_anomaly_scores_y_values, [0] * self.get_length(), self.get_probationary_index(), self.get_length(), gaussian_window_size)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        end = time.time()

        if save_final:
            # Saving whole models (architecture + weights + optimizer state)
            model.save("save_rnn/" + self.name + "_final_rnn.h5")

        if plot_anomaly_score:
            plt.scatter(error_anomaly_scores_x_values, error_anomaly_scores_y_values)
            # plt.title("Data dependent anomaly scores")
            plt.savefig("dmkd_plot.eps", bbox_inches='tight')
            plt.show()

            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    # https://github.com/twitter/AnomalyDetection/blob/master/R/ts_anom_detection.R
    def twitterad(self, period, max_anoms=.1, direction='pos', alpha=.05,
                  plot_anomaly_score=False):
        """
        Input:

        max_anoms =  Maximum number of anomalies that S-H-ESD will
        detect as a percentage of the data. Default is 10 percent

        direction = Directionality of the anomalies to be detected.
        Options are: 'pos', 'neg', 'both'. Default is 'pos'

        alpha = The level of statistical significance with which
        to accept or reject anomalies. Default is .05

        period = number of time steps in a season
        period must be >1. i.e. if there is no seasonality, you cannot use this method

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        Output:

        dictionary:
        anomaly scores, time
        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        outlier_indices_df = use_r_twitterad(self.dataframe, self.main_variable_name, max_anoms,
                                             direction, alpha, period)

        if "x" not in outlier_indices_df:
            end = time.time()
            return {"Anomaly Scores": [0] * self.get_length(),
                    "Time": end - start}
        outlier_indices = outlier_indices_df["x"].values
        anomaly_scores = convert_outlier_index(self.get_length(), outlier_indices)
        end = time.time()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    def surus(self, freq=1, autodiff=True, forcediff=False, scale=True, lpenalty="default", spenalty="default", plot_anomaly_score=False):
        """
        Input:

        freq = seasonality size (can be 1 if no seasonality present)

        autodiff = boolean, True -> use ADF to determine if differencing is
        needed to make ts stationary, default is True

        forcediff = boolean, True -> always compute differences,
        default is False

        scale = boolean, True -> normalize the ts to 0 mean and
        1 variance, default is True

        lpenalty = scalar -> thresholding for low rank approximation of ts,
        default is 1

        spenalty = scalar -> thresholding for separating noise and
        sparse outliers, default is 1.4 / sqrt(max(frequency, ifelse(is.data.frame(X), nrow(X), length(X)) / frequency))

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        Output:

        dictionary:
        anomaly scores, time
        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        outlier_indices = use_r_netflixsurus(self.dataframe, self.main_variable_name, freq, autodiff, forcediff, scale, lpenalty, spenalty)
        anomaly_scores = convert_outlier_index(self.get_length(), outlier_indices)
        end = time.time()
        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()
        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    def hotsax(self, win_size=100, num_discords=2,
               a_size=3, paa_size=3, z_threshold=0.01,
               plot_anomaly_score=False):
        """
        Input:

        win_size = sliding window size

        num_discords = how many discords do you want to find?

        a_size = alphabet size (how many letters can we use to represent the ts?)

        paa_size = piecewise aggregate approximation size

        z_threshold = z-normalization threshold, only relevant for normalization and not for Confidence intervals

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        Output:

        dictionary:
        anomaly scores, time

        word of warning: outlier indices return the first index
        of a discord which is a subsequence (len win_size) that is different
        """
        if self.missing:
            raise ValueError("Missing time steps. Cannot use this method.")
        start = time.time()
        discords = find_discords_hotsax(self.dataframe[self.main_variable_name].values,
                                        win_size=win_size, num_discords=num_discords,
                                        a_size=a_size, paa_size=paa_size,
                                        z_threshold=z_threshold)
        outlier_indices = []
        for discord in discords:
            outlier_indices.append(discord[0])
        anomaly_scores = convert_outlier_index(self.get_length(), outlier_indices)
        end = time.time()
        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()
        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    def donut(self, gaussian_window_size, window_size=120, plot_reconstruction_prob=False, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        window size for donut

        plot_reconstruction_prob = True or False
        Whether or not to plot the reconstruction probabilities

        plot_anomaly_score  =  True or False
        Whether or not to plot the anomaly score (normalized)
        Output:

        list of outlier indices, time, anomaly scores

        """
        # this method can deal with missing time steps
        start = time.time()
        timestamp = self.dataframe["timestamp"].values
        values = self.dataframe[self.main_variable_name].values
        labels = np.zeros_like(values, dtype=np.int32)

        # warning: missing time stamps should not exist in df
        # i.e. you do not want a row like: timestamp, NaN
        # just make that row disappear!
        # i.e. just use self.dataframe and NOT
        # self.dataframe_missing
        # warning: if there are duplicate time stamps,
        # complete_timstamp will fail
        # so use remove_duplicate_time_steps()
        # see https://github.com/haowen-xu/donut/blob/master/donut/preprocessing.py
        # line 38
        timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

        # probationary period is first 15 percent and is for training
        test_portion = .85
        test_n = int(len(values) * test_portion)
        train_values, test_values = values[:-test_n], values[-test_n:]
        train_labels, _ = labels[:-test_n], labels[-test_n:]
        train_missing, test_missing = missing[:-test_n], missing[-test_n:]
        train_values, mean, std = standardize_kpi(train_values, excludes=np.logical_or(train_labels, train_missing))
        test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

        with tf.variable_scope('model') as model_vs:
            model = Donut(
                # hidden layer for p_theta(x|z)
                h_for_p_x=Sq([
                    keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001),
                                       activation=tf.nn.relu),
                    keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001),
                                       activation=tf.nn.relu),
                ]),
                # hidden layer for q_theta(z|x)
                h_for_q_z=Sq([
                    keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001),
                                       activation=tf.nn.relu),
                    keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001),
                                       activation=tf.nn.relu),
                ]),
                # x_dims = window size (authors use 120)
                x_dims=window_size,
                # z_dims = K in the paper = latent dimension (3 to 8 work well)
                z_dims=5,
            )

        trainer = DonutTrainer(model=model, model_vs=model_vs)
        predictor = DonutPredictor(model)

        with tf.Session().as_default():
            # train only on train set (probationary period)
            trainer.fit(train_values, train_labels, train_missing, mean, std)
            # score on train set AND test set
            # to prevent anomaly scores from being weird
            # right after probationary period
            all_score = predictor.get_score(list(train_values) + list(test_values), list(train_missing) + list(test_missing))

        # documentation of get_score function:
        # Get the `reconstruction probability` of specified KPI observations.
        # The larger `reconstruction probability`, the less likely a point
        # is anomaly.  You may take the negative of the score, if you want
        # something to directly indicate the severity of anomaly.

        negative_test_score = [-score for score in all_score]
        anomaly_scores = [0] * (window_size - 1) + negative_test_score
        anomaly_scores = determine_anomaly_scores_error(anomaly_scores, [0] * len(anomaly_scores), self.get_probationary_index(), self.get_length(), gaussian_window_size)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        end = time.time()

        if plot_reconstruction_prob:
            plt.plot(negative_test_score)
            plt.title("reconstruction Probability")
            plt.show()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    def stl_resid(self, gaussian_window_size, n_periods, swindow, outer, fill_option="linear", plot_resid=False, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        n_periods  = number of periods per season (for stl)
        must be at least 4

        swindow = window width (for stl)

        outer = number of outer iterations (for stl)

        fill_option = fill option for stl residuals if missing is true

        plot_resid = True or False. Whether or not to plot STL residuals and ACF of it

        plot_anomaly_score  =  True or False
        Whether or not to plot the anomaly score (normalized)

        Output:

        dictionary:
        anomaly scores, time
        """
        # this method can deal with missing time steps
        if n_periods < 4:
            raise ValueError("n_periods must be at least 4.")
        start = time.time()
        if self.missing:
            # raw and stl_remainder have NaNs still.
            # use R's imputeTS to fill in NaNs in
            # stl residuals
            raw, stl_remainder = use_r_stl(self.dataframe_missing, self.main_variable_name, n_periods, swindow, outer, self.missing, fill_option=fill_option)
            self.dataframe_missing["stl remainder of value"] = stl_remainder
        else:
            raw, stl_remainder = use_r_stl(self.dataframe, self.main_variable_name, n_periods, swindow, outer, False, fill_option=fill_option)
            self.dataframe["stl remainder of value"] = stl_remainder

        anomaly_scores = determine_anomaly_scores_error(stl_remainder, [0] * self.get_length(), self.get_probationary_index(), self.get_length(), gaussian_window_size)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        end = time.time()

        if plot_resid:
            if not self.missing:
                sm.graphics.tsa.plot_acf(self.dataframe[self.main_variable_name].values, lags=100)
                plt.title("ACF of Data")
                plt.show()
            plt.plot(stl_remainder)
            plt.title("STL Data remainder")
            plt.show()
            sm.graphics.tsa.plot_acf(stl_remainder, lags=100)
            plt.title("ACF of STL remainder")
            plt.show()
        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()
        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}


class ContextualAnomalyDetection:

    def __init__(self, dataframe, timestep, dateformat, main_variable_name, contextual_variable_names, name, missing=False):
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format=dateformat)
        self.dataframe = dataframe
        self.timestep = timestep
        self.dateformat = dateformat
        self.main_variable_name = main_variable_name
        self.contextual_variable_names = contextual_variable_names
        self.missing = missing
        self.name = name

        if ("timestamp" not in dataframe):
            raise ValueError("The given dataframe must have 'timestamp' columns.")

        # no time steps are allowed to be missing
        # i.e. there should be a row in the df for EVERY time step
        if len(find_missing_time_steps(dataframe["timestamp"].values[0], dataframe["timestamp"].values[-1], timestep, dataframe)) > 0:
            raise ValueError("There are missing time steps. Rows for every time step must exist, contextual variables cannot have NaNs, altho main variables can have NaNs. See fill_missing_time_steps.")

        # contextual variables cannot have NaNs
        for contextual_variable in self.contextual_variable_names:
            if self.dataframe[contextual_variable].isnull().values.any():
                raise ValueError("Contextual variables cannot have NaNs. See fill_missing_time_steps.")

        # main variables can have NaNs
        # but this restricts the AD methods that can be used
        if self.dataframe[self.main_variable_name].isnull().values.any():
            print("WARNING: The main variable has NaNs. This restricts the kinds of anomaly detection methods that can be used.")
            self.missing = True

    def get_length(self):
        """Get length of dataframe"""
        return len(self.dataframe)

    def get_probationary_index(self):
        """Get the index of the first time step after the probationary period"""
        return int(.15 * self.get_length())

    def get_mean(self):
        """Get mean of dataframe"""
        return np.mean(self.dataframe[self.main_variable_name])

    def get_variance(self):
        """Get variance of dataframe"""
        return np.var(self.dataframe[self.main_variable_name])

    def get_std(self):
        """Get standard deviation of dataframe"""
        return np.std(self.dataframe[self.main_variable_name])

    def get_timestep(self):
        """Get time step size of dataframe"""
        return self.timestep

    def get_dateformat(self):
        """Get the string date format of dataframe"""
        return self.dateformat

    def convert_true_outlier_date(self, outlier_date_list):
        """
        Input:

        List of true outlier dates
        e.g. ["2011-07-26 06:00:01"]

        Output:
        Indices of those true outlier dates

        Also makes an outlier column for the dataframe
        """
        outlier_index_list = np.zeros(len(self.dataframe))
        only_1s = []
        for date in outlier_date_list:
            outlier_index = self.dataframe.loc[self.dataframe['timestamp'] == date].index[0]
            outlier_index_list[outlier_index] = 1
            only_1s.append(outlier_index)
        self.dataframe["outlier"] = outlier_index_list
        return only_1s

    def plot_outliers(self, predicted_outlier_index_values, true_outlier_index_values):
        # data
        plt.plot(self.dataframe.index.values, self.dataframe[self.main_variable_name].values, color="blue", alpha=.5)
        # plt.scatter(self.dataframe.index.values, self.dataframe[self.main_variable_name].values, color="blue", alpha=.5)

        plt.axvline(self.get_probationary_index(), color="black", label="probationary line")

        # predicted outliers
        if predicted_outlier_index_values:
            outlier_y_values = []
            for outlier_index in predicted_outlier_index_values:
                outlier_y_values.append(self.dataframe[self.main_variable_name].values[outlier_index])
            plt.scatter(predicted_outlier_index_values, outlier_y_values, color="green", label="predicted outliers", alpha=.5)

        # true outliers
        outlier_y_values = []
        for outlier_index in true_outlier_index_values:
            outlier_y_values.append(self.dataframe[self.main_variable_name].values[outlier_index])
        plt.scatter(true_outlier_index_values, outlier_y_values, color="red", label="true outliers", marker="*", alpha=.5, zorder=100)
        # plt.legend()
        plt.savefig("dmkd_plot.eps", bbox_inches='tight')
        plt.show()

    # You can either provide p,d,q,P,D,Q,s yourself or
    # use autoarima to have them determined for you,
    # By default, s is 0 (no seasonality).
    # If there is seasonality, regardless of autoarima or not,
    # you should set s appropriately by using an ACF plot.
    # Warning: Statsmodels' SARIMAX represents lack of
    # seasonality using s = 0, but autoarima represents
    # lack of seasonality using s = 1.
    def sarimax(self, gaussian_window_size, p=1, d=1, q=1, big_p=0, big_d=0,
                big_q=0, s=0, autoarima=False,
                autoarima_season=True, maxiter=50,
                plot_acf=False, plot_anomaly_score=False):
        """
        Input:

        gaussian_window_size = window size for scoring

        SARIMA parameters (p,d,q)x(P,D,Q,s).

        autoarima = True or False.
        Whether or not to use auto.arima
        to determine parameters. If True, the given
        parameters with the exception of s are ignored
        and determined using autoarima

        autoarima_season = True or False.
        Whether or not to allow seasonal behavior
        in autoarima. If True, allows for P,D,Q,s

        maxiter = maximum number of iterations

        plot_acf = True or False. Whether or not to plot acf and pacf

        plot_anomaly_score = True or False.
        Whether or not to plot the anomaly score

        Output:

        dictionary:
        anomaly scores, time, RMSE
        """
        if self.missing:
            raise ValueError("NaNs in main variable. Cannot use this method.")
        start = time.time()

        # use autoarima to determine p,d,q,etc
        if autoarima:
            # seasonality allowed
            # determine p,d,q,P,D,Q,s
            if autoarima_season:
                arima = pm.auto_arima(self.dataframe[self.main_variable_name].values[0:self.get_probationary_index()], error_action='ignore', trace=1, seasonal=True, m=s, maxiter=maxiter)
                p, d, q = arima.order
                big_p, big_d, big_q, s = arima.seasonal_order
            # seasonality not allowed
            # determine only p,d,q
            else:
                arima = pm.auto_arima(self.dataframe[self.main_variable_name].values[0:self.get_probationary_index()], error_action='ignore', trace=1, maxiter=maxiter)
                p, d, q = arima.order

        if len(self.dataframe) < det_req_obs(p, d, q, big_p, big_d, big_q, s):
            raise ValueError("Not enough observations for given parameters.")

        exog_data = []
        for i in range(len(self.dataframe)):
            exog_data_i = []
            for cv in self.contextual_variable_names:
                exog_data_i.append(self.dataframe[cv].values[i])
            exog_data.append(exog_data_i)

        print("Final parameters: ", (p, d, q, big_p, big_d, big_q, s))

        sarimax_model = sm.tsa.SARIMAX(self.dataframe[self.main_variable_name].values, exog=exog_data, order=(p, d, q), seasonal_order=(big_p, big_d, big_q, s))
        sarimax_model_results = sarimax_model.fit(disp=0)
        sarimax_model_forecast = sarimax_model_results.get_prediction(exog=exog_data, full_results=True, alpha=0.05)

        predictions = sarimax_model_forecast.predicted_mean
        actual = self.dataframe[self.main_variable_name].values

        anomaly_scores = determine_anomaly_scores_error(actual, predictions, self.get_probationary_index(), self.get_length(), gaussian_window_size)

        end = time.time()

        if plot_acf:

            sm.graphics.tsa.plot_acf(self.dataframe[self.main_variable_name].values, lags=100)
            plt.title("Data ACF")
            plt.show()

            sm.graphics.tsa.plot_pacf(self.dataframe[self.main_variable_name].values, lags=100)
            plt.title("Data PACF")
            plt.show()

            sm.graphics.tsa.plot_acf(sarimax_model_results.resid, lags=100)
            plt.title("Residual ACF after fitting to SARIMAX")
            plt.show()

            sm.graphics.tsa.plot_pacf(sarimax_model_results.resid, lags=100)
            plt.title("Residual PACF after fitting to SARIMAX")
            plt.show()

        if plot_anomaly_score:
            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly Scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start,
                "RMSE": rmse(predictions, actual)}

    def rnn(self, gaussian_window_size, n_lag, n_seq, n_test, n_epochs, n_batch, n_neurons, training_index, plot_anomaly_score=False, load=False, save_final=True):
        """
        Input:

        gaussian_window_size = window size for scoring

        n_lag = number of observations as input

        n_seq = number of observations as output

        n_test = the last n_test time steps form the test set

        n_epochs = number of epochs

        n_batch = batch size

        n_neurons = number of cells in each layer

        training_index = to separate training set from rest of set for determining anomaly scores

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        load = True or False. Whether or not to load a pretrained model

        save_final = True or False. Whether or not to save the final model (even after batch training)

        Output:

        dictionary:
        anomaly scores, time
        """
        if self.missing:
            raise ValueError("NaNs in main variable. Cannot use this method.")

        if training_index + n_lag + n_seq > len(self.dataframe):
            raise ValueError("training_index + n_lag + n_seq must be less than the length of the dataframe to produce anomaly scores.")

        start = time.time()
        # all variables will be scaled
        indexed_date_count_df = pd.DataFrame({})
        indexed_date_count_df[self.main_variable_name] = normalize_data(self.dataframe, self.main_variable_name)
        for cv in self.contextual_variable_names:
            indexed_date_count_df[cv] = normalize_data(self.dataframe, cv)
        my_data = indexed_date_count_df[0:training_index]

        # number of columns for train and test is
        # (n_lag + n_seq) * dimensions

        # number of rows for train is
        # training_index-n_lag-n_test-n_seq+1
        # number of rows for test is n_test

        train, test = prepare_data(my_data, n_test, n_lag, n_seq, False)
        dimensions = len(self.contextual_variable_names) + 1

        if not load:
            model = fit_rnn(train, n_lag, n_seq, n_batch, n_epochs, n_neurons, dimensions)
            # Saving/loading whole models (architecture + weights + optimizer state)
            model.save("save_rnn/" + self.name + "_initial_rnn.h5")
        else:
            print("WARNING: printed time does not include initial training time")
            model = load_model("save_rnn/" + self.name + "_initial_rnn.h5")

        start_point = 0
        error_anomaly_scores_x_values = []
        error_anomaly_scores_y_values = []

        while start_point + n_lag + n_seq < len(indexed_date_count_df):
            end_point = start_point + n_lag + n_seq
            batch_data = indexed_date_count_df[start_point:end_point]

            # train_batch shape is 1 x (n_lag + n_seq) * dimensions
            train_batch = prepare_data(batch_data, n_test, n_lag, n_seq, True)

            x, y = train_batch[:, 0:n_lag * dimensions], train_batch[:, n_lag * dimensions:]

            # x goes from (1,n_lag*dimensions) to (1,1,n_lag*dimensions)  after reshape
            x = x.reshape(x.shape[0], 1, x.shape[1])
            model.train_on_batch(x, y)
            predictions = model.predict_on_batch(x)

            # x has length n_lag*dimensions
            # predictions and y are vectors of length n_seq*dimensions

            anomaly_score = det_anomaly_score(y, predictions, dimensions)
            error_anomaly_scores_x_values.append(end_point)
            error_anomaly_scores_y_values.append(anomaly_score)
            start_point += 1

        padded_error_anomaly_scores_y_values = [0] * (self.get_length() - len(error_anomaly_scores_y_values)) + error_anomaly_scores_y_values

        # save these unnormalized scores! then for various gaussian window sizes
        # you can just use those scores
        save_path = "rnn_unnormalized_anomaly_scores/" + self.name + "_rnn_unnormalized_scores"
        end = time.time()
        joblib.dump({"Unnormalized Anomaly Scores": padded_error_anomaly_scores_y_values, "Time": end - start}, save_path)

        # these scores are normalized using the gaussian window size
        anomaly_scores = determine_anomaly_scores_error(padded_error_anomaly_scores_y_values, [0] * self.get_length(), self.get_probationary_index(), self.get_length(), gaussian_window_size)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        end = time.time()

        if save_final:
            # Saving whole models (architecture + weights + optimizer state)
            model.save("save_rnn/" + self.name + "_final_rnn.h5")

        if plot_anomaly_score:
            plt.scatter(error_anomaly_scores_x_values, error_anomaly_scores_y_values)
            plt.title("Data dependent anomaly scores")
            plt.show()

            plt.plot(anomaly_scores)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()

        return {"Anomaly Scores": anomaly_scores,
                "Time": end - start}

    def liu(self, n_periods, swindow, outer, fill_option="linear", plot_resid=False, plot_anomaly_score=False):
        """
        Input:

        n_periods  = number of periods per season (for stl)
        must be at least 4

        swindow = window width (for stl)

        outer = number of outer iterations (for stl)

        fill_option = "linear" for linear interpolation, "spline" for spline, "stine" for Stineman interpolation

        plot_resid = True or False. Whether or not to plot STL residuals and ACF of it

        plot_anomaly_score = True or False. Whether or not to plot data with anomaly scores

        Output:

        list of outlier indices, time, anomaly scores
        """
        # can deal with NaNs for main variable
        if n_periods < 4:
            raise ValueError("n_periods must be at least 4.")
        start = time.time()
        # raw and stl_remainder can have NaNs still.
        # this is problematic for layer two
        # so use_r_stl will include a fill which will fill in the stl residuals
        # using R imputeTS library
        raw, stl_remainder = use_r_stl(self.dataframe, self.main_variable_name, n_periods, swindow, outer, self.missing, fill_option=fill_option)

        self.dataframe["stl remainder of " + self.main_variable_name] = stl_remainder

        z = normalize_data(self.dataframe, "stl remainder of " + self.main_variable_name)
        x = make_context_variables_x(self.contextual_variable_names, self.dataframe)

        v, mus, theta_squareds = step_two(self.contextual_variable_names, z, x)

        # v is len(self.dataframe) - 1 bc
        # v[i] = probability of deviation score at i is way higher than
        # the deviation score at i-1
        # i.e. there is no anomaly score for the first step

        end = time.time()

        if plot_resid:
            if not self.missing:
                sm.graphics.tsa.plot_acf(self.dataframe[self.main_variable_name].values, lags=100)
                plt.title("ACF of Data")
                plt.show()
            plt.plot(stl_remainder)
            plt.title("STL remainder")
            plt.show()
            sm.graphics.tsa.plot_acf(stl_remainder, lags=100)
            plt.title("ACF of STL remainder")
            plt.show()

        if plot_anomaly_score:

            plt.plot([0] + v)
            plt.axvline(self.get_probationary_index(), color="black", label="probationary line")
            plt.title("Anomaly scores")
            plt.show()

        return {"Anomaly Scores": [0] + v,
                "Time": end - start}
