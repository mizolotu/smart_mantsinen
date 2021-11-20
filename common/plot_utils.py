import plotly.graph_objs as go
import numpy as np

from matplotlib import pyplot as pp

def moving_average(x, step=1, window=10):
    seq = []
    n = x.shape[0]
    for i in np.arange(0, n, step):
        idx = np.arange(np.maximum(0, i - window), np.minimum(n - 1, i + window + 1))
        seq.append(np.mean(x[idx, :], axis=0))
    return np.vstack(seq)

def plot_line(x, y, marker, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def plot_multiple_lines(xs, ys, markers, xlabel, ylabel, fpath):
    pp.figure(figsize=(16, 12))
    for x, y, marker in zip(xs, ys, markers):
        pp.plot(x, y, marker)
    pp.xlabel(xlabel, fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

def generate_line_scatter(names, values, colors, xlabel, ylabel, show_legend=True, xrange=None):

    traces = []

    for i in range(len(names)):
        x = values[i][0].tolist()
        y = values[i][1].tolist()

        if xrange is None:
            xrange = [x[0], x[-1]]

        traces.append(
            go.Scatter(
                x=x,
                y=y,
                line=dict(color=colors[i]),
                mode='lines',
                showlegend=show_legend,
                name=names[i],
            )
        )

    layout = go.Layout(
        template='plotly_white',
        xaxis=dict(
            title=xlabel,
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False,
            range=xrange
        ),
        yaxis=dict(
            title=ylabel,
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False
        ),
    )

    return traces, layout