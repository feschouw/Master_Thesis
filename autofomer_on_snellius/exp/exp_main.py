import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, FEDformer, Linear
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from analysis.decoder_lens import model_forward_decoderlens

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Informer": Informer,
            "Reformer": Reformer,
            "FEDformer": FEDformer,
            "Linear": Linear,
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.from_checkpoint:
            path = os.path.join(self.args.checkpoints, self.args.from_checkpoint)
            model_path = path + "/" + "checkpoint.pth"
            model.load_state_dict(torch.load(model_path))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        empty_data=False,
        empty_time=False,
    ):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        if empty_data:  # zero is close to the mean of data (0.01), so good baseline
            dec_inp = (
                torch.cat([dec_inp[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )
            batch_x = torch.zeros_like(batch_x)
        else:
            dec_inp = (
                torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )

        if empty_time:  # replace by mean time
            batch_x_mark = torch.zeros_like(batch_x_mark)
            batch_y_mark = torch.zeros_like(batch_y_mark)

        # encoder - decoder

        def _run_model():
            if "Linear" in self.args.model:
                outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

        return outputs, batch_y

    def _predict_decoderlens(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        empty_data=False,
        empty_time=False,
    ):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        if empty_data:  # zero is close to the mean of data (0.01), so good baseline
            dec_inp = (
                torch.cat([dec_inp[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )
            batch_x = torch.zeros_like(batch_x)
        else:
            dec_inp = (
                torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )

        if empty_time:  # replace by mean time
            batch_x_mark = torch.zeros_like(batch_x_mark)
            batch_y_mark = torch.zeros_like(batch_y_mark)

        # encoder - decoder

        def _run_model():
            if "Linear" in self.args.model:
                pass
                # outputs = self.model(batch_x) # not implemented
            else:
                outputs = model_forward_decoderlens(
                    self.model,
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    tmp_layernorm=True,
                )
            final_output = outputs[0]
            all_outputs = outputs[-1]
            return final_output, all_outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                final_output, all_outputs = _run_model()
        else:
            final_output, all_outputs = _run_model()

        f_dim = -1 if self.args.features == "MS" else 0

        outputs = final_output[:, -self.args.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

        return outputs, batch_y, all_outputs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)

                if self.args.random_future:
                    batch_y = (
                        torch.rand(batch_y.shape) * 2 - 1
                    )  # random values between -1 and 1

                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_overview = {}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                if self.args.random_future:
                    batch_y = (
                        torch.rand(batch_y.shape) * 2 - 1
                    )  # random values between -1 and 1

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            epoch_overview[epoch] = {
                "train": train_loss,
                "val": vali_loss,
                "test": test_loss,
            }

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + "epoch_overview.npy", epoch_overview)
        print(f"Saved epoch_oveview here: {folder_path + 'epoch_overview.npy'}")

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        if (
            self.args.HourOfDay_offset != 0
            or self.args.DayOfWeek_offset != 0
            or self.args.DayOfMonth_offset != 0
            or self.args.DayOfYear_offset != 0
            or self.args.empty_data
            or self.args.empty_time
        ):
            test_setting = "{}_{}_{}_{}{}{}{}hod_{}dow_{}dom_{}doy_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                self.args.model_id,
                self.args.model,
                self.args.data,
                "random_future_" if self.args.random_future else "",
                "empty_data_" if self.args.empty_data else "",
                "empty_time_" if self.args.empty_time else "",
                self.args.HourOfDay_offset,
                self.args.DayOfWeek_offset,
                self.args.DayOfMonth_offset,
                self.args.DayOfYear_offset,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des,
                0,
            )
        else:
            test_setting = setting

        folder_path = "./test_results/" + test_setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.decoder_lens_test:
            all_preds = defaultdict(list)
        else:
            preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)

                if self.args.random_future:
                    batch_y = (
                        torch.rand(batch_y.shape) * 2 - 1
                    )  # random values between -1 and 1

                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if not self.args.decoder_lens_test:

                    outputs, batch_y = self._predict(
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark,
                        self.args.empty_data,
                        self.args.empty_time,
                    )

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
                else:
                    outputs, batch_y, all_outputs = self._predict_decoderlens(
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark,
                        self.args.empty_data,
                        self.args.empty_time,
                    )

                    batch_y = batch_y.detach().cpu().numpy()

                    for layer, tmp_out in enumerate(all_outputs):
                        tmp_out = tmp_out.detach().cpu().numpy()
                        all_preds[layer].append(tmp_out)

                    true = batch_y
                    trues.append(true)

                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        vis_preds = {}
                        for layer, tmp_out in enumerate(all_outputs):
                            tmp_out = tmp_out.detach().cpu().numpy()
                            pd = np.concatenate(
                                (
                                    [np.nan for i in range(batch_x.shape[1])],
                                    tmp_out[0, :, -1],
                                ),
                                axis=0,
                            )
                            vis_preds[f"{layer+1}/{len(all_outputs)}"] = pd
                            plt.plot(
                                pd, label=f"prediction {layer+1}/{len(all_outputs)}"
                            )
                        visual(
                            gt, vis_preds, os.path.join(folder_path, str(i) + ".pdf")
                        )

        if not self.args.decoder_lens_test:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print("test shape:", preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print("test shape:", preds.shape, trues.shape)

            # result save
            folder_path = "./results/" + test_setting + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, mase = metric(preds, trues)
            print("mse:{}, mae:{}, mase:{}".format(mse, mae, mase))
            f = open("result.txt", "a")
            f.write(test_setting + "  \n")
            f.write("mse:{}, mae:{}, mase: {}".format(mse, mae, mase))
            f.write("\n")
            f.write("\n")
            f.close()

            np.save(
                folder_path + "metrics.npy",
                np.array([mae, mse, rmse, mape, mspe, mase]),
            )
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        else:
            max_layer = max(all_preds.keys())
            for layer in all_preds.keys():
                all_preds[layer] = np.concatenate(all_preds[layer], axis=0)
            trues = np.concatenate(trues, axis=0)
            print("test shape:", all_preds[max_layer].shape, trues.shape)
            for layer in all_preds.keys():
                all_preds[layer] = all_preds[layer].reshape(
                    -1, all_preds[layer].shape[-2], all_preds[layer].shape[-1]
                )
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print("test shape:", all_preds[max_layer].shape, trues.shape)

            # result save
            folder_path = "./results/" + test_setting + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            metrics = {}
            metrics["mae"] = {}
            metrics["mse"] = {}
            metrics["rmse"] = {}
            metrics["mape"] = {}
            metrics["mspe"] = {}
            metrics["mase"] = {}

            for layer, preds in all_preds.items():
                mae, mse, rmse, mape, mspe, mase = metric(preds, trues)
                metrics["mae"][layer] = mae
                metrics["mse"][layer] = mse
                metrics["rmse"][layer] = rmse
                metrics["mape"][layer] = mape
                metrics["mspe"][layer] = mspe
                metrics["mase"][layer] = mase

                print(
                    "layer {} - mse:{}, mae:{}, mase:{}".format(layer, mse, mae, mase)
                )
                f = open("result.txt", "a")
                f.write(test_setting + "  \n")
                f.write("mse:{}, mae:{}, mase: {}".format(mse, mae, mase))
                f.write("\n")
                f.write("\n")
                f.close()

            np.save(folder_path + "metrics.npy", metrics)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
