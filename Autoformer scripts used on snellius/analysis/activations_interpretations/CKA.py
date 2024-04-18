import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append("..")

import analysis.activations_interpretations.CKA_utils as CKA_utils
import activations_interpretations.data_utils as data_utils
import activations_interpretations.activations as activations
from collections import defaultdict

label_len = 48


def obtain_CKA_heatmaps_layerwise(
    activations1, activations2, modelnames, type="linear", headwise=True
):
    # Obtain CKA heatmaps for activations per layer
    # Only activations2 is allowed to have just activations of just one head
    results = defaultdict(defaultdict)
    descriptions = defaultdict(defaultdict)
    for component in activations1[-1].keys():
        for layer in activations1[-1][component].keys():
            X_list = tuple(
                X[component][layer][0].squeeze().numpy() for X in activations1
            )
            X = np.concatenate(X_list, axis=0)
            if len(X.shape) == 3:
                if headwise:
                    n_heads_X = X.shape[1]
                    for head in range(n_heads_X):
                        results[component][
                            "layer" + str(layer) + "_head" + str(head)
                        ] = []
                        descriptions[component][
                            "layer" + str(layer) + "_head" + str(head)
                        ] = []
            else:
                results[component]["layer" + str(layer)] = []
                descriptions[component]["layer" + str(layer)] = []
            for comparison_layer in activations2[-1][component].keys():
                Y_list = tuple(
                    Y[component][comparison_layer][0].squeeze().numpy()
                    for Y in activations2
                )
                Y = np.concatenate(Y_list, axis=0)
                if len(Y.shape) == 3:
                    n_heads_Y = Y.shape[1]
                    one_head_Y = False
                else:
                    one_head_Y = True

                if len(X.shape) == 3:  # we have activations per head
                    if headwise:
                        for head_x in range(n_heads_X):
                            X_i = X[:, head_x, :]
                            if one_head_Y:
                                Y_i = Y
                                if type == "linear":
                                    score = CKA_utils.linear_CKA(X_i, Y_i)
                                elif type == "kernel":
                                    score = CKA_utils.kernel_CKA(X_i, Y_i)

                                results[component][
                                    "layer" + str(layer) + "_head" + str(head_x)
                                ].append(score)
                                descriptions[component][
                                    "layer" + str(layer) + "_head" + str(head_x)
                                ].append("layer" + str(comparison_layer) + "_head0")
                            else:
                                for head_y in range(n_heads_Y):
                                    Y_i = Y[:, head_y, :]

                                    if type == "linear":
                                        score = CKA_utils.linear_CKA(X_i, Y_i)
                                    elif type == "kernel":
                                        score = CKA_utils.kernel_CKA(X_i, Y_i)

                                    results[component][
                                        "layer" + str(layer) + "_head" + str(head_x)
                                    ].append(score)
                                    descriptions[component][
                                        "layer" + str(layer) + "_head" + str(head_x)
                                    ].append(
                                        "layer"
                                        + str(comparison_layer)
                                        + "_head"
                                        + str(head_y)
                                    )

                else:
                    if type == "linear":
                        score = CKA_utils.linear_CKA(X, Y)
                    elif type == "kernel":
                        score = CKA_utils.kernel_CKA(X, Y)

                    results[component]["layer" + str(layer)].append(score)
                    descriptions[component]["layer" + str(layer)].append(
                        "layer" + str(comparison_layer)
                    )
        d = results[component]
        results[component] = dict(sorted(d.items(), reverse=True))
    return results, descriptions, modelnames


def obtain_CKA_heatmaps_layerwise_handcrafted(
    activations1, handcrafted_reprs, modelnames, type="linear", headwise=True
):
    """
    Obtain CKA heatmaps for activations with handcrafted representations, this method compares all components of activations1 with all components of activations2
    """
    results = defaultdict(defaultdict)
    descriptions = defaultdict(defaultdict)
    for component in activations1[-1].keys():
        for layer in activations1[-1][component].keys():

            X_list = tuple(
                X[component][layer][0].squeeze().numpy() for X in activations1
            )
            X = np.concatenate(X_list, axis=0)
            if len(X.shape) == 3:
                if headwise:
                    n_heads_X = X.shape[1]
                    for head in range(n_heads_X):
                        results[component][
                            "layer" + str(layer) + "_head" + str(head)
                        ] = []
                        descriptions[component][
                            "layer" + str(layer) + "_head" + str(head)
                        ] = []
            else:
                results[component]["layer" + str(layer)] = []
                descriptions[component]["layer" + str(layer)] = []
            for comparison_component in handcrafted_reprs[-1].keys():
                Y_list = tuple(Y[comparison_component] for Y in handcrafted_reprs)
                Y = np.concatenate(Y_list, axis=0)
                if len(Y.shape) == 1:
                    Y = np.expand_dims(Y, axis=1)

                if len(X.shape) == 3:  # we have activations per head
                    if headwise:
                        for head_x in range(n_heads_X):
                            X_i = X[:, head_x, :]

                            if type == "linear":
                                score = CKA_utils.linear_CKA(X_i, Y)
                            elif type == "kernel":
                                score = CKA_utils.kernel_CKA(X_i, Y)

                            results[component][
                                "layer" + str(layer) + "_head" + str(head_x)
                            ].append(score)
                            descriptions[component][
                                "layer" + str(layer) + "_head" + str(head_x)
                            ].append(str(comparison_component))

                else:
                    if type == "linear":
                        score = CKA_utils.linear_CKA(X, Y)
                    elif type == "kernel":
                        score = CKA_utils.kernel_CKA(X, Y)

                    results[component]["layer" + str(layer)].append(score)
                    descriptions[component]["layer" + str(layer)].append(
                        str(comparison_component)
                    )
        d = results[component]
        results[component] = dict(sorted(d.items(), reverse=True))
    return results, descriptions, modelnames


def plot_CKA_heatmaps(
    scores,
    descriptions,
    modelnames,
    type="linear",
    save_fig=False,
    name_fig="",
    width=4,
):
    for component in scores.keys():
        if (
            len(scores[component].values()) != 0
        ):  # if some hooks are not yet implemented, can be removed later

            fig, ax = plt.subplots(figsize=(width, 6))

            im = ax.imshow(scores[component].values(), aspect="auto")

            ax.set_title(f"{component} ({type} CKA)")
            fig.colorbar(im, ax=ax)

            ax.set_xticks(
                ticks=range(len(list(descriptions[component].values())[0])),
                labels=list(descriptions[component].values())[0],
                rotation=90,
            )

            yticks_labels = list(scores[component].keys())

            if "_" in yticks_labels[0]:
                split_strings = [string.split("_") for string in yticks_labels]

                layer_labels = set([parts[0] for parts in split_strings])
                head_labels = set([parts[1] for parts in split_strings])

                ax.yaxis.set_minor_locator(
                    plt.FixedLocator(np.arange(len(yticks_labels)))
                )
                ax.yaxis.set_minor_formatter(plt.NullFormatter())

                major_ticks = (
                    (np.arange(len(layer_labels))) * len(head_labels)
                    + len(head_labels)
                    - 1
                )
                ax.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
                ax.yaxis.set_major_formatter(
                    plt.FixedFormatter(
                        [parts[0] for parts in split_strings][:: len(head_labels)]
                    )
                )

            else:
                ax.set_yticks(
                    ticks=range(len(yticks_labels)),
                    labels=yticks_labels,
                )

            ax.set_ylabel(modelnames[0])
            ax.set_xlabel(modelnames[1])

            if save_fig:
                if not os.path.exists(
                    f"plots/{modelnames[0]}_{modelnames[1]}_{name_fig}"
                ):
                    os.makedirs(f"plots/{modelnames[0]}_{modelnames[1]}_{name_fig}")
                plt.savefig(
                    f"plots/{modelnames[0]}_{modelnames[1]}_{name_fig}/{modelnames[0]}_{modelnames[1]}_{name_fig}_{component}.png",
                    bbox_inches="tight",
                )

            plt.show()


def CKA_main(
    modelA,
    modelB,
    data_loader,
    nameA="modelA",
    nameB="modelB",
    CKA_type="linear",
    num_stimuli=1,
    headwise=True,
    save_fig=False,
    name_fig="",
):
    stimuli = []

    for i, stimulus in enumerate(data_loader):
        if i == num_stimuli:
            break
        stimuli.append(stimulus)
        seq_x, seq_y, seq_x_mark, seq_y_mark = stimulus
        plt.plot(seq_x[0, :, -1])
        plt.suptitle(f"stimulus {i}")
        if save_fig:
            if not os.path.exists(
                f"plots/{nameA}_{nameB}_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}"
            ):
                os.makedirs(
                    f"plots/{nameA}_{nameB}_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}"
                )
            plt.savefig(
                f"plots/{nameA}_{nameB}_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}/{nameA}_{nameB}{'_' if name_fig != '' else ''}{name_fig}_stimulus {i}.png",
                bbox_inches="tight",
            )
        plt.show()
        data_utils.obtain_plot_predictions(
            pred_len=96,
            data_set=nameA,
            data_loader=data_loader,
            model=modelA,
            stimulus=stimulus,
            to_save=save_fig,
            name_fig=f"plots/{nameA}_{nameB}_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}/{nameA}{'_' if name_fig != '' else ''}{name_fig}_stimulus {i}.png",
        )
        data_utils.obtain_plot_predictions(
            pred_len=96,
            data_set=nameB,
            data_loader=data_loader,
            model=modelB,
            stimulus=stimulus,
            to_save=save_fig,
            name_fig=f"plots/{nameA}_{nameB}_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}/{nameB}{'_' if name_fig != '' else ''}{name_fig}_stimulus {i}.png",
        )

    activationsA = activations.get_activations_for_dataset(
        model=modelA, stimuli=stimuli
    )
    activationsB = activations.get_activations_for_dataset(
        model=modelB, stimuli=stimuli
    )
    # activationsA = get_activations_for_dataset(model=modelA, stimuli=[stimulus])
    # activationsB = get_activations_for_dataset(model=modelB, stimuli=[stimulus])

    scores, descriptions, modelnames = obtain_CKA_heatmaps_layerwise(
        activationsA,
        activationsB,
        modelnames=[nameA, nameB],
        type=CKA_type,
        headwise=headwise,
    )

    plot_CKA_heatmaps(
        scores,
        descriptions,
        modelnames,
        type=CKA_type,
        save_fig=save_fig,
        name_fig=f"{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}",
    )


def CKA_decompose(
    model,
    data_loader,
    name="model",
    CKA_type="linear",
    num_stimuli=1,
    headwise=True,
    save_fig=False,
    name_fig="",
):
    stimuli = []

    for i, stimulus in enumerate(data_loader):
        if i == num_stimuli:
            break
        stimuli.append(stimulus)
        seq_x, seq_y, seq_x_mark, seq_y_mark = stimulus
        plt.plot(seq_x[0, :, -1])
        plt.suptitle(f"stimulus {i}")
        if save_fig:
            if not os.path.exists(
                f"plots/{name}_decompose_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}"
            ):
                os.makedirs(
                    f"plots/{name}_decompose_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}"
                )
            plt.savefig(
                f"plots/{name}_decompose_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}/{name}{'_' if name_fig != '' else ''}{name_fig}_stimulus {i}.png",
                bbox_inches="tight",
            )
        plt.show()
        data_utils.obtain_plot_predictions(
            pred_len=96,
            data_set=name,
            data_loader=data_loader,
            model=model,
            stimulus=stimulus,
            to_save=save_fig,
            name_fig=f"plots/{name}_decompose_{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}/{name}{'_' if name_fig != '' else ''}{name_fig}_stimulus {i}.png",
        )

    activationsA = activations.get_activations_for_dataset(model=model, stimuli=stimuli)

    activationsB = activations.get_decompose_activations(stimuli=stimuli)

    scores, descriptions, modelnames = obtain_CKA_heatmaps_layerwise_handcrafted(
        activationsA,
        activationsB,
        modelnames=[name, "decompose"],
        type=CKA_type,
        headwise=headwise,
    )

    plot_CKA_heatmaps(
        scores,
        descriptions,
        modelnames,
        type=CKA_type,
        save_fig=save_fig,
        name_fig=f"{num_stimuli}stimuli{'_' if name_fig != '' else ''}{name_fig}",
    )


if __name__ == "__main__":

    autoformer_7 = data_utils.obtain_autoformer(pred_len=96, dataset="sinus7")
    data_loader_7 = data_utils.obtain_data_loader(pred_len=96, dataset="sinus7")

    data_utils.obtain_plot_predictions(
        96,
        "sinus7",
        data_loader_7,
        autoformer_7,
        name_plot="test",
        item=0,
        alpha_gt=0.5,
        alpha_pred=1,
    )

    CKA_main(
        autoformer_7,
        autoformer_7,
        data_loader_7,
        "autoformer 7",
        "autoformer 7",
        num_stimuli=1,
    )
