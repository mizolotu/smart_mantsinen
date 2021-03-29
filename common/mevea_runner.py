import numpy as np
import gym, pyautogui, cv2

from common.runners import AbstractEnvRunner, swap_and_flatten
from common.solver_utils import get_solver_path, start_solver, stop_solver
from common.server_utils import is_backend_registered, delete_id
from time import sleep, time
from threading import Thread

class MeveaRunner(AbstractEnvRunner):

    def __init__(self, *, env, model, n_steps, gamma, lam, debug=False):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.solver_path = get_solver_path()
        self.mvs = env.mvs
        self.server = env.server
        self.recording = False
        self.debug = debug

        # reset environments in debug mode

        if self.debug:
            str = input('id:\n')
            self.backend_ids = [int(item) for item in str.split(',')]

        # init data

        self.mb_obs = [[] for _ in range(self.n_envs)]
        self.mb_actions = [[] for _ in range(self.n_envs)]
        self.mb_values = [[] for _ in range(self.n_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs)]
        self.mb_dones = [[] for _ in range(self.n_envs)]
        self.mb_rewards = [[] for _ in range(self.n_envs)]
        self.scores = [[] for _ in range(self.n_envs)]

    def _start(self, headless=True, sleep_interval=1):
        self.backend_procs = []
        for mvs, server in zip(self.mvs, self.server):
            proc = start_solver(self.solver_path, mvs, headless=headless)
            self.backend_procs.append(proc)
            while not is_backend_registered(server, proc.pid):
                sleep(sleep_interval)

    def record(self, video_file, sleep_interval=0.05, x=210, y=90, width=755, height=400):
        screen_size = pyautogui.Size(width, height)
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(video_file, fourcc, 20.0, (screen_size))
        while self.recording:
            img = pyautogui.screenshot(region=(x, y, width, height))
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            sleep(sleep_interval)
        cv2.destroyAllWindows()
        out.release()

    def _stop(self):
        for proc, server in zip(self.backend_procs, self.server):
            stop_solver(proc)
            delete_id(server, proc.pid)

    def _run_all(self, video_file=None, headless=True):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        # start environment right here

        self._start(headless=headless)
        self.env.set_attr('id', [proc.pid for proc in self.backend_procs])
        self.obs[:] = self.env.reset()

        if video_file is not None:
            self.recording = True
            thr = Thread(target=self.record, args=(video_file,), daemon=True)
            thr.start()

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        scores = [[] for _ in range(self.n_envs)]
        scores_1 = [[] for _ in range(self.n_envs)]
        scores_2 = [[] for _ in range(self.n_envs)]
        scores_3 = [[] for _ in range(self.n_envs)]
        tstep = 0
        for _ in range(self.n_steps):

            tstart = time()

            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions

            # Clip the actions to avoid out of bound error

            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for ri, r in enumerate(rewards):
                scores[ri].append(r)
            for ri, inf in enumerate(infos):
                scores_1[ri].append(infos[ri]['rc1'])
                scores_2[ri].append(infos[ri]['rc2'])
                scores_3[ri].append(infos[ri]['rc3'])

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:

                # Abort training early

                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False

                    # Return dummy values

                    return [None] * 9
            mb_rewards.append(rewards)

            tstep += (time() - tstart)

        print('Step time: {0}'.format(tstep / self.n_steps))

        # stop recording

        self.recording = False

        # stop backends

        self._stop()

        # gather info

        for escore, escore1, escore2, escore3 in zip(scores, scores_1, scores_2, scores_3):
            maybe_ep_info = {'r': np.mean(escore), 'rc1': np.mean(escore1), 'rc2': np.mean(escore2), 'rc3': np.mean(escore3)}
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn

        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def _run_one(self, env_idx):

        tstart = time()

        for _ in range(self.n_steps):

            if self.debug:
                print(_)

            # step model

            actions, values, self.states, neglogpacs = self.model.step(self.obs[env_idx:env_idx+1], self.states, self.dones[env_idx:env_idx+1])

            # save results

            self.mb_obs[env_idx].append(self.obs.copy()[env_idx])
            self.mb_actions[env_idx].append(actions[0])
            self.mb_values[env_idx].append(values[0])
            self.mb_neglogpacs[env_idx].append(neglogpacs[0])
            self.mb_dones[env_idx].append(self.dones[env_idx])

            # Clip the actions to avoid out of bound error

            clipped_actions = actions
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            tnow = time()

            self.obs[env_idx], rewards, self.dones[env_idx], infos = self.env.step_one(env_idx, clipped_actions)

            if self.debug:
                print('Env', env_idx, 'step takes', time() - tnow, 'seconds')

            self.scores[env_idx].append([rewards, infos['rc1'], infos['rc2'], infos['rc3']])

            self.model.num_timesteps += 1

            if self.callback is not None:

                # Abort training early

                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False

                    # Return dummy values

                    return [None] * 9

            self.mb_rewards[env_idx].append(rewards)

        print('Step time: {0}'.format((time() - tstart) / self.n_steps))

        stop_solver(self.backend_procs[env_idx])
        delete_id(self.server[env_idx], self.backend_procs[env_idx].pid)

    def _run(self, video_file=None, headless=False):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        self.mb_obs = [[] for _ in range(self.n_envs)]
        self.mb_actions = [[] for _ in range(self.n_envs)]
        self.mb_values = [[] for _ in range(self.n_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs)]
        self.mb_dones = [[] for _ in range(self.n_envs)]
        self.mb_rewards = [[] for _ in range(self.n_envs)]

        ep_infos = []

        # start environment's backend

        if self.debug:
            self.env.set_attr('id', self.backend_ids)
        else:
            self._start(headless=headless)
            self.env.set_attr('id', [proc.pid for proc in self.backend_procs])

        # reset environment's frontend

        self.obs[:] = self.env.reset()

        # start video recording

        if video_file is not None:
            self.recording = True
            thr = Thread(target=self.record, args=(video_file,), daemon=True)
            thr.start()

        # run steps in different threads



        threads = []
        for env_idx in range(self.n_envs):
            th = Thread(target=self._run_one, args=(env_idx,))
            th.start()
            threads.append(th)
        [th.join() for th in threads]

        # stop recording

        self.recording = False

        # combine data gathered into batches

        mb_obs = [np.vstack([self.mb_obs[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_rewards = [np.hstack([self.mb_rewards[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_actions = [np.vstack([self.mb_actions[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_values = [np.hstack([self.mb_values[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_neglogpacs = [np.hstack([self.mb_neglogpacs[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_dones = [np.hstack([self.mb_dones[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_scores = [np.vstack([self.scores[idx][step] for idx in range(self.n_envs)]) for step in range(self.n_steps)]
        mb_states = self.states
        self.dones = np.array(self.dones)

        for scores_in_env in mb_scores:
            maybe_ep_info = {
                'r': np.mean(scores_in_env[:, 0]),
                'rc1': np.mean(scores_in_env[:, 1]),
                'rc2': np.mean(scores_in_env[:, 2]),
                'rc3': np.mean(scores_in_env[:, 3])
            }
            ep_infos.append(maybe_ep_info)

        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn

        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        # reset data

        self.mb_obs = [[] for _ in range(self.n_envs)]
        self.mb_actions = [[] for _ in range(self.n_envs)]
        self.mb_values = [[] for _ in range(self.n_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs)]
        self.mb_dones = [[] for _ in range(self.n_envs)]
        self.mb_rewards = [[] for _ in range(self.n_envs)]
        self.scores = [[] for _ in range(self.n_envs)]

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward