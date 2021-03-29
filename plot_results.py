import os, pandas
import os.path as osp
import argparse as arp
import plotly.io as pio
import plotly.graph_objs as go

from common.plot_utils import generate_line_scatter, moving_average

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot results')
    parser.add_argument('-i', '--input', help='Input directory', default='models/mevea/mantsinen/ppo')
    parser.add_argument('-o', '--output', help='Output directory', default='figures/mevea/mantsinen/ppo')
    args = parser.parse_args()

    score_keys = ['ep_reward_mean', 'ep_reward_c1_mean', 'ep_reward_c2_mean']
    score_labels = ['Total score', 'Score for following the path', 'Score for reaching the target']
    score_names = ['total_score', 'path_score', 'target_score']
    some_colors = ['rgb(64,120,211)', 'rgb(0,100,80)', 'rgb(237,2,11)']
    more_colors = ['rgb(255,165,0)', 'rgb(139,0,139)', 'rgb(0,51,102)']

    fname = osp.join(args.input, 'progress.csv')
    p = pandas.read_csv(fname, delimiter=',', dtype=float)
    x = p['total_timesteps'].values
    xlabel = 'Timesteps'

    for color, key, label, name in zip(some_colors, score_keys, score_labels, score_names):
        y = moving_average(p[key].values.reshape(-1, 1), window=1)[:, 0]
        data = [[x, y]]
        colors = [color]
        names = [label]
        ylabel = label

        traces, layout = generate_line_scatter(names, data, colors, xlabel, ylabel, xrange=[0, 4358144])

        # save results

        ftypes = ['png', 'pdf']
        if not osp.exists(args.output):
            os.mkdir(args.output)
        fig_fname = '{0}/{1}'.format(args.output, name)
        fig = go.Figure(data=traces, layout=layout)
        for ftype in ftypes:
            pio.write_image(fig, '{0}.{1}'.format(fig_fname, ftype))