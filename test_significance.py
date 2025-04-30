from scipy.stats import t
from scipy import stats
import pandas as pd
import numpy as np
import argparse


def test_significance(ytrue, ypred, yconpred):
    n = len(ytrue)
    yfinalpred_res_abs = np.absolute(np.array(ytrue) - np.array(ypred))
    yconpred_res_abs = np.absolute(np.array(ytrue) - np.array(yconpred))
    yfcra_diff = yfinalpred_res_abs - yconpred_res_abs
    yfcra_diff_mean, yfcra_sd = np.mean(yfcra_diff), np.std(yfcra_diff)
    yfcra_diff_t = yfcra_diff_mean / (yfcra_sd / np.sqrt(n))
    yfcra_diff_p = t.sf(np.abs(yfcra_diff_t), n-1)
    return yfcra_diff_p


def test_model(embeds_group1, embeds_group2):
    res = []
    for i in range(len(embeds_group1)):
        df1 = pd.read_csv(embeds_group1[i], skiprows=1)
        df2 = pd.read_csv(embeds_group2[i], skiprows=1)
        preds_col, gt_col = df1.columns[1:]
        p = test_significance(df1[gt_col], df1[preds_col], df2[preds_col])
        res.append(p)
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test significance of model predictions.')
    parser.add_argument('--embeds1_path', type=str, help='Path to the first group of embeddings.')
    parser.add_argument('--embeds2_path', type=str, help='Path to the second group of embeddings.')
    args = parser.parse_args()

    embeds_group1 = []
    embeds_group2 = []

    if args.embeds1_path is not None and args.embeds1_path is not None:
        embeds_group1 = [args.embeds1_path]
        embeds_group2 = [args.embeds2_path]

    p_values = test_model(embeds_group1, embeds_group2)

    print(p_values)