model_dir = 'C:\\Users\\mevea\\MeveaModels'
model_path = f'{model_dir}\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs'  # <- change this if needed
server = 'http://127.0.0.1:5000'
model_output = 'models/mevea/mantsinen/'
video_output = 'videos/mevea/mantsinen/'
img_output = 'screenshots/mevea/mantsinen/'
trajectory_dir = 'data/trajectories'
trajectory_plot_dir = 'data/trajectory_plots'
signal_dir = 'data/signals'
dataset_dir = 'data/dataset'
waypoints_dir = 'data/waypoints'
default_actions = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 100.0, 0.0]]
nenvs = 8  # <- change this if needed
nsteps = 4096
batch_size = 256
npretrain = 10000
patience = 100
learning_rate = 2.5e-4
ntrain = 10000000000
sleep_interval = 3
use_inputs = True
use_outputs = True
action_scale = 1
wp_size = 1
lookback = 256
tstep = 0.01
bonus = 10
nwaypoints = 16
validation_size = 0.1
ppo_net_arch = [
    #('conv1d', 256, 10, 5, 'valid'), ('conv1d', 512, 10, 5, 'valid'), ('dense', 1024),
    #('conv1d', 256, 10, 5, 'valid'), ('lstm', 256, False), ('dense', 512),
    #('lstm', 64, True), ('lstm', 64, False), ('dense', 64),
    ('lstm', 64, False), ('dense', 64),
    dict(vf=[64, 64]), dict(pi=[64, 64])
]