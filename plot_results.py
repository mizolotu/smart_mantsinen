import os, pandas
import os.path as osp
import argparse as arp
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

from common.plot_utils import generate_line_scatter, moving_average

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot results')
    parser.add_argument('-i', '--input', help='Input directory', default='models/mevea/mantsinen/ppo')
    parser.add_argument('-o', '--output', help='Output directory', default='figures/mevea/mantsinen/ppo')
    parser.add_argument('-s', '--steps', help='Steps', default=20e6)
    parser.add_argument('-w', '--window', help='Moving average window', default=100)
    args = parser.parse_args()

    #score_keys = ['ep_reward_mean', 'ep_path_dist_mean', 'ep_target_dist_mean']
    score_keys = ['ep_reward_mean', 'policy_loss', 'value_loss', 'std']
    #score_labels = ['Total score', 'Mean distance to the path', 'Mean distance to the target']
    score_labels = ['Total score', 'Policy loss', 'Value loss', 'Actions std']
    #score_names = ['total_score', 'path_error', 'target_error']
    score_names = ['total_score', 'policy_loss', 'value_loss', 'std']
    some_colors = ['rgb(64,120,211)', 'rgb(0,100,80)', 'rgb(237,2,11)', 'rgb(255,165,0)']
    more_colors = ['rgb(139,0,139)', 'rgb(0,51,102)']

    fname = osp.join(args.input, 'progress.csv')
    p = pandas.read_csv(fname, delimiter=',', dtype=float)
    non_nan_idx = np.where(np.isnan(np.sum(p.values, axis=1)) == False)[0]
    x = p['total_timesteps'].values[non_nan_idx]
    #x = p['value_loss'].values
    xlabel = 'Timesteps'
    xlimit = args.steps if args.steps is not None else x[-1]
    #xlimit = (args.steps // (nenvs * nsteps) - args.window + 1) * nsteps * nenvs

    for color, key, label, name in zip(some_colors, score_keys, score_labels, score_names):
        y = moving_average(p[key].values[non_nan_idx].reshape(-1, 1), window=args.window)[:, 0]
        #data = [[x[args.window-1:] - nenvs * nsteps * (args.window - 1), y[args.window-1:]]]
        data = [[x, y]]
        colors = [color]
        names = [label]
        ylabel = label

        traces, layout = generate_line_scatter(names, data, colors, xlabel, ylabel, xrange=[0, xlimit])

        # save results

        ftypes = ['png', 'pdf']
        if not osp.exists(args.output):
            os.mkdir(args.output)
        fig_fname = '{0}/{1}'.format(args.output, name)
        fig = go.Figure(data=traces, layout=layout)
        for ftype in ftypes:
            pio.write_image(fig, '{0}.{1}'.format(fig_fname, ftype))