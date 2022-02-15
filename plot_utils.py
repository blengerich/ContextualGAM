import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def axvlines(xs, min_height=0.3, max_height=0.4, color='orange', **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = (min_height, max_height, np.nan) #plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims)[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, color=color, alpha=0.6, scaley = False, **plot_kwargs)
    return plot

def center_and_scatter(xs, ys, with_var=False, plot_args=None):
    order = np.argsort(xs)
    for y_ar in ys:
        y_ar -= np.min(y_ar)
    y_means = np.mean(ys, axis=0)
    plt.plot(xs[order], y_means[order], **plot_args)
    if with_var:
        y_mins = y_means - 2*np.std(ys, axis=0)
        y_maxes = y_means + 2*np.std(ys, axis=0)
        plt.fill_between(xs[order], y_mins[order], y_maxes[order], alpha=0.2)

def plot_cgams_homogeneous_effects(nams, C_train, X_all_train, ylabel="Addition to P(Progression) (Log-Odds)"):
    p = C_train.shape[-1]
    for j, feat in enumerate(C_train.columns):
        fig = plt.figure(figsize=(8, 6))
        xs = C_train[feat].values
        empty = np.zeros_like(C_train)
        empty[:, j] = xs
        empty_all = np.zeros_like(X_all_train)
        j_all = X_all_train.columns.tolist().index(feat)
        empty_all[:, j_all] = xs
        ys = np.zeros((len(nams), len(C_train)))
        for k, nam in enumerate(nams):
            try:
                ys[k] += nam.base_model.predict_proba(empty_all)[:, 1].squeeze()
            except:
                pass
            try:
                ys[k] += nam.skip_encoder_model(empty).numpy().squeeze()
            except:
                pass
        center_and_scatter(xs, ys, with_var=True, plot_args={'marker': '+'})
        plt.xlabel(feat, fontsize=32)
        plt.ylim([0, 2])
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

def plot_cgams_heterogeneous(cgams, C_train, X_train):
    p = C_train.shape[-1]
    empty = np.zeros_like(C_train)
    for i, outcome in enumerate(X_train.columns):
        ys = []
        for j, feat in enumerate(C_train.columns):
            xs = sorted(C_train[feat].values)
            xs_orig = xs
            x_min = np.percentile(xs_orig, 1)
            x_max = np.percentile(xs_orig, 99)
            empty[:, j] = xs
            ys.append(np.array([cgam.encoder(empty).numpy()[:, i] for cgam in cgams]))
            for y in ys[-1]:
                y_min = np.min(y[np.logical_and(xs_orig >= x_min, xs_orig <= x_max)])
                y -= y_min
            empty[:, j] = 0.
            
        ys = np.array(ys) 
        means = np.mean(ys, axis=-1)
        mean_effect = np.mean(means, axis=-1)
        overall_std = np.std(means, axis=-1)
        for j in range(len(ys)):
            ys[j] += mean_effect[j]
        
        for j, feat in enumerate(C_train.columns):
            xs = sorted(C_train[feat].values)
            xs_orig = np.array(xs)
            x_min = np.percentile(xs_orig, 1)
            x_max = np.percentile(xs_orig, 99)
            fig = plt.figure(figsize=(8, 6))
            on_idxs = X_train[outcome].values > 0
            off_idxs = X_train[outcome].values <= 0
            if len(set(xs)) == 2: # Plot Boolean
                y_means = np.mean(ys[j], axis=0)
                y_stds  = np.std(ys[j], axis=0)
                plt.bar([0.2, 0.8], [y_means[0], y_means[-1]],
                        yerr=[2*y_stds[0], 2*y_stds[-1]], color='blue',
                       ecolor='black', capsize=10, width=0.4, align='center', alpha=0.5)
                plt.xticks([0.2, 0.8], ["No", "Yes"])
                axvlines(C_train[feat].loc[on_idxs]/1.6 + 0.2 + \
                                 np.random.uniform(-0.2, 0.2, size=(np.sum(on_idxs))), 
                     min_height=-0.06, max_height=-0.02, color='red')
                axvlines(C_train[feat].loc[off_idxs]/1.6 + 0.175 + \
                                np.random.uniform(-0.2, 0.2, size=(np.sum(off_idxs))),
                     min_height=-0.06, max_height=-0.02, color='black')
            else: # Plot Continuous-valued
                window = np.logical_and(xs_orig >= x_min, xs_orig <= x_max)
                plt.plot(xs_orig[window], np.mean(ys[j], axis=0)[window], color='blue')
                plt.fill_between(xs_orig[window],
                                 np.percentile(ys[j], 2.5, axis=0)[window] - overall_std[j],
                                 np.percentile(ys[j], 97.5, axis=0)[window] + overall_std[j],
                                 alpha=0.2)
                axvlines(C_train[feat].loc[on_idxs] + \
                                 np.random.uniform(-0.1, 0.1, size=(np.sum(on_idxs))), 
                     min_height=-0.06, max_height=-0.02, color='red')
                axvlines(C_train[feat].loc[off_idxs] + \
                                np.random.uniform(-0.1, 0.1, size=(np.sum(off_idxs))),
                     min_height=-0.06, max_height=-0.02, color='black')

            plt.xlabel(feat, fontsize=30)
            plt.ylabel("Impact on P(Y=1) (Log-Odds)".format(outcome), fontsize=22)
            plt.title(outcome, fontsize=24)
            plt.xticks(fontsize=24)
            plt.xlim(x_min, x_max)
            plt.tight_layout()
            plt.show()
            