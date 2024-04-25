import torch
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import defaultdict

sys.path.append("..")

from collections import defaultdict

label_len = 48


def get_activations_for_dataset(model, stimuli: list, pred_len=96):
    # Create a list of dictionaries to store the activations of each sample.
    activations = []

    def reset_activations():
        # Reset the activations for a new sample.
        activations.append(
            {
                "value_vectors": defaultdict(list),
                "individual_heads": defaultdict(list),
                "attention_weights": defaultdict(list),
                "attention.query_projection": defaultdict(list),
                "attention.key_projection": defaultdict(list),
                "attention.value_projection": defaultdict(list),
                "attention.out_projection": defaultdict(list),
                # "conv2": defaultdict(list),
            }
        )

    def value_vectors_hook(layer_id):
        def hook(module, input, output):
            activations[-1]["value_vectors"][layer_id].append(output.detach())

        return hook

    def attention_hook(layer_id):
        def hook(module, input, output):
            activations[-1]["individual_heads"][layer_id].append(output.detach())

        return hook

    def attention_weights_hook(layer_id):
        def hook(module, input, output):
            activations[-1]["attention_weights"][layer_id].append(output.detach())

        return hook

    def attention_query_projection(layer_id):
        def hook(module, input, output):
            activations[-1]["attention.query_projection"][layer_id].append(
                output.detach()
            )

        return hook

    def attention_key_projection(layer_id):
        def hook(module, input, output):
            activations[-1]["attention.key_projection"][layer_id].append(
                output.detach()
            )

        return hook

    def attention_value_projection(layer_id):
        def hook(module, input, output):
            activations[-1]["attention.value_projection"][layer_id].append(
                output.detach()
            )

        return hook

    def attention_out_projection(layer_id):
        def hook(module, input, output):
            activations[-1]["attention.out_projection"][layer_id].append(
                output.detach()
            )

        return hook

    def conv2_hook(layer_id):
        def hook(module, input, output):
            activations[-1]["conv2"][layer_id].append(output.detach())

        return hook

    # Register the hooks.
    for i, layer in enumerate(model.encoder.attn_layers):
        layer.attention.value_vectors.register_forward_hook(value_vectors_hook(i))
        layer.attention.individual_heads.register_forward_hook(attention_hook(i))
        layer.attention.attn_weights.register_forward_hook(attention_weights_hook(i))
        layer.attention.key_projection.register_forward_hook(
            attention_key_projection(i)
        )
        layer.attention.value_projection.register_forward_hook(
            attention_value_projection(i)
        )
        layer.attention.query_projection.register_forward_hook(
            attention_query_projection(i)
        )
        layer.attention.out_projection.register_forward_hook(
            attention_out_projection(i)
        )
        # layer.conv2.register_forward_hook(conv2_hook(i))

    # Go through your dataset and compute the forward passes.
    # res = []
    for (
        seq_x,
        seq_y,
        seq_x_mark,
        seq_y_mark,
    ) in stimuli:  # stimulus as returned by dataloader
        reset_activations()
        torch.manual_seed(42)

        # decoder input
        dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).float()
        outputs_autoformer = model(
            seq_x.float(), seq_x_mark.float(), dec_inp, seq_y_mark.float()
        )

    # Unregister the hooks to make sure they don't interfere with the next dataset.
    for i, layer in enumerate(model.encoder.attn_layers):
        layer.attention.value_vectors._forward_hooks.clear()
        layer.attention.individual_heads._forward_hooks.clear()
        layer.attention.attn_weights._forward_hooks.clear()
        layer.attention.key_projection._forward_hooks.clear()
        layer.attention.value_projection._forward_hooks.clear()
        layer.attention.query_projection._forward_hooks.clear()
        layer.attention.out_projection._forward_hooks.clear()
        # layer.conv2._forward_hooks.clear()

    return activations


def get_decompose_activations(stimuli, to_plot=False):
    """
    stimuli: list of (seq_x, seq_y, seq_x_mark, seq_y_mark), where seq_x has shape (1, num_datapoints, num_timeseries)
    returns (num_datapoints, num_timeseries)
    """
    activations = []

    for seq_x, _, _, _ in stimuli:
        activations.append(
            {
                "trend": None,
                "seasonal": None,
                "residual": None,
            }
        )
        if to_plot:
            plt.plot(seq_x[0, :, -1])
        decompose_result = seasonal_decompose(
            seq_x.squeeze(), model="additive", period=25, extrapolate_trend="freq"
        )  # moving avg of 25 is used in series decomposition
        trend, seasonal, residual = (
            decompose_result.trend,
            decompose_result.seasonal,
            decompose_result.resid,
        )
        if to_plot:
            plt.plot(seq_x[0, :, -1])
            plt.plot(trend[:, -1])
            plt.show()
            plt.plot(seasonal[:, -1], label="seasonal")
            plt.plot(residual[:, -1], label="residual")
            plt.legend()
            plt.show()

        activations[-1]["trend"] = trend
        activations[-1]["seas_and_res"] = seasonal + residual
        activations[-1]["seasonal"] = seasonal
        activations[-1]["residual"] = residual

    return activations


def get_timestamp_activations(stimuli):
    """
    stimuli: list of (seq_x, seq_y, seq_x_mark, seq_y_mark), where seq_x has shape (1, num_datapoints, num_timeseries)
    returns (num_datapoints, num_timeseries)
    """
    activations = []

    for seq_x, seq_y, seq_x_mark, seq_y_mark in stimuli:

        # this only works with a freq of 'h'
        activations.append(
            {
                "hourofday": None,
                "dayofweek": None,
                "dayofmonth": None,
                "dayofyear": None,
                "all_timestamp": None,
            }
        )

        seq_x_mark = seq_x_mark.squeeze()
        activations[-1]["hourofday"] = seq_x_mark[:, 0]
        activations[-1]["dayofweek"] = seq_x_mark[:, 1]
        activations[-1]["dayofmonth"] = seq_x_mark[:, 2]
        activations[-1]["dayofyear"] = seq_x_mark[:, 3]
        activations[-1]["all_timestamp"] = seq_x_mark

    return activations
