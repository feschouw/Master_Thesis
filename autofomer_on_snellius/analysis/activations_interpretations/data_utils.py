# utils for interpreting activations of autoformer networks

import sys

sys.path.append("..")
sys.path.append("../..")

from models import Autoformer
from utils.tools import dotdict
from utils.timefeatures import time_features_from_frequency_str_uncalled
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


import torch
import numpy as np
import matplotlib.pyplot as plt


def obtain_autoformer(pred_len, dataset, el=2, nh=8):
    # dataset = [sinusoidal]
    assert pred_len in [96, 192, 336, 720]

    args = dotdict()
    args.pred_len = pred_len

    args.label_len = 48

    args.target = "OT"
    args.des = "train"
    args.dropout = 0.05
    args.num_workers = 10
    args.gpu = 0
    args.lradj = "type1"
    args.devices = "0"
    args.use_gpu = False
    args.use_multi_gpu = False
    args.freq = "h"
    args.checkpoints = "./checkpoints/"
    args.bucket_size = 4
    args.n_hashes = 4
    args.is_trainging = True
    args.data = "custom"
    args.seq_len = 96
    # args.label_len = 48
    args.e_layers = el
    args.d_layers = 1
    args.n_heads = nh
    args.d_model = 512
    args.des = "Exp"
    args.itr = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 1
    args.distil = True
    args.output_attention = True  # in order to save hooks
    args.embed = "timeF"

    if "ECL" in dataset:
        args.factor = 3
        args.enc_in = 321
        args.dec_in = 321
        args.c_out = 321
        args.features = "M"
    else:
        args.factor = 1
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.features = "S"

    autoformer_path = f"/Users/angelavansprang/Documents/PhD/transformers for time series/Interpreting Time Series Transformers/checkpoints/{dataset}_96_{pred_len}_Autoformer_{'ETTm2' if dataset == 'ETTm2' else 'custom'}_ft{args.features}_sl96_ll{'96' if dataset == 'ETTm2' else '48'}_pl{pred_len}_dm512_nh{nh}_el{el}_dl1_df2048_fc{args.factor}_ebtimeF_dtTrue_Exp_0/checkpoint.pth"

    autoformer = Autoformer.Model(args).float()
    autoformer.load_state_dict(
        torch.load(autoformer_path, map_location=torch.device("cpu"))
    )

    autoformer.eval()

    return autoformer


def obtain_data_loader(pred_len, dataset, seq_len=96, split="train"):
    label_len = 48

    if dataset == "ECL" or dataset == "electricity":
        datafolder = "electricity"
        features = "M"
    else:
        datafolder = "generated"
        features = "S"

    dataset = Dataset_Custom(
        root_path=f"/Users/angelavansprang/Documents/PhD/transformers for time series/Interpreting Time Series Transformers/dataset/{datafolder}/",
        data_path=f"{dataset}.csv",
        flag=split,  # flag must be train in order for the linear dataloader to contain the same data
        size=[seq_len, label_len, pred_len],  # seq_len, label_len, pred_len
        features=features,
        target="OT",  # default
        timeenc=1,
        freq="h",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    return dataloader


def day_of_year_to_date(day_of_year):
    # Define the number of days in each month
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # # Check for leap year
    # is_leap_year = lambda year: (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # Initialize month and day
    month = 1
    day = int(day_of_year)

    # Update day and month based on the number of days in each month
    while day > days_in_month[month - 1]:
        day -= days_in_month[month - 1]
        # if month == 2 and is_leap_year(year):
        #     day -= 1
        month += 1

    # Define month names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Return the date
    return f"{month_names[month - 1]} {day}"


def obtain_timestamps(seq_mark, time_feat):
    daysofweek = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    timestamps = []

    seq_mark = seq_mark.squeeze()
    for i, feat in enumerate(time_feat):
        seq_mark[:, i] = feat.inverse(seq_mark[:, i])
    for timestamp in range(seq_mark.shape[0]):
        time = f"{round(seq_mark[timestamp,0].item())}H"
        weekday = daysofweek[round(seq_mark[timestamp, 1].item())]
        date = day_of_year_to_date(seq_mark[timestamp, 3])
        timestamps.append(f"{time} {weekday} {date}")
    return timestamps


def obtain_plot_predictions(
    pred_len,
    data_set,
    data_loader,
    model,
    stimulus=None,
    name_plot="",
    to_save=False,
    name_fig="",
    label_len=48,
    item=0,
    alpha_gt=1,
    alpha_pred=0.5,
    vlines=[],
    timestamps=False,
):
    with torch.no_grad():
        for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(data_loader):
            if i == item:
                # seq_x is batch_size x seq_len x N_timeseries
                # seq_y is batch_size x (label_len + pred_len) x N_timeseries
                # seq_x_mark is batch_size x seq_len X N_timefeatures
                # seq_y_mark is batch_size x (label_len+pred_len) X N_timefeatures
                if stimulus is not None:
                    seq_x, seq_y, seq_x_mark, seq_y_mark = stimulus

                # decoder input
                dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).float()

                # gt = np.concatenate((seq_x[0, :, 0], seq_y[0, -pred_len:, 0]), axis=0)
                gt = np.concatenate((seq_x[0, :, -1], seq_y[0, -pred_len:, -1]), axis=0)

                outputs_autoformer, _ = model(
                    seq_x.float(), seq_x_mark.float(), dec_inp, seq_y_mark.float()
                )
                pd_autoformer = np.concatenate(
                    (
                        [np.nan for i in range(seq_x.shape[1])],
                        outputs_autoformer[0, -pred_len:, -1],
                    ),
                    axis=0,
                )

                plt.plot(gt, label="ground truth", color="black", alpha=alpha_gt)
                plt.plot(pd_autoformer, label="autoformer", alpha=alpha_pred)

                for vline in vlines:
                    plt.axvline(x=vline, color="b", linestyle="--")

                plt.legend()
                plt.xlabel("Time")
                plt.ylabel("Scaled time series")
                plt.suptitle(data_set)

                if timestamps:
                    time_feat = time_features_from_frequency_str_uncalled("h")
                    x_timestamps = obtain_timestamps(seq_x_mark, time_feat)
                    y_timestamps = obtain_timestamps(seq_y_mark, time_feat)[-pred_len:]
                    timestamps = x_timestamps + y_timestamps

                    tick_positions = [
                        i
                        for i, tick_label in enumerate(timestamps)
                        if tick_label.startswith("0H")
                    ]
                    tick_labels = [timestamps[i][3:] for i in tick_positions]
                    plt.xticks(tick_positions, tick_labels, rotation=45)
                    plt.grid(True)

                if to_save:
                    # plt.savefig(f"plots/{name_plot}.png")
                    plt.savefig(name_fig, bbox_inches="tight")
                plt.show()
