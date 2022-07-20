import gym
import pygame
import matplotlib
import argparse
from gym import logger

try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

from collections import deque
from pygame.locals import VIDEORESIZE

import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
import resource
import time

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

import numpy as np
import ruamel.yaml as yaml
import torch

# this allows common import
# sys.path.append(str(pathlib.Path(__file__).parent))
# sys.path.append(str(pathlib.Path(__file__).parent.parent))

sys.path.append(str(pathlib.Path(".").parent))
sys.path.append(str(pathlib.Path(".").parent.parent))

import agent
import elements
import common


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


configs = pathlib.Path(sys.argv[0]).parent / 'configs.yaml'
configs = yaml.safe_load(configs.read_text())
config = elements.Config(configs['defaults'])
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
    config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)

logdir = pathlib.Path(config.logdir).expanduser()

config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)

seed=271

def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = common.DMC(task, config.action_repeat, config.image_size)
        env = common.NormalizeAction(env)
    elif suite == 'atari':
        env = common.Atari(
            task, config.action_repeat, config.image_size, config.grayscale,
            life_done=False, sticky_actions=True, all_actions=True, seed=seed)
        env = common.OneHotAction(env)
    elif suite == 'retro':
        env = common.Retro(task, config.action_repeat, config.image_size, config.grayscale,
                           life_done=False, sticky_actions=True, all_actions=True, seed=seed)
        env = common.OneHotAction(env)

    elif suite == 'minigrid':
        env = common.Minigrid(task, config.image_size, config.grayscale, seed=seed)
        env = common.OneHotAction(env)

        env.get_keys_to_action = lambda : {():6,(119,):2,(97,):0,(100,):1,(101,):3,(114,):4,(116,):5}#,(102,):6}

    else:
        raise NotImplementedError(suite)

    if tuple(config.encoder['keys']) == ('flatten',):
        assert tuple(config.decoder['keys']) == ('flatten',), "config: decoder is not flatten"
        env = common.FlattenImageObs(env)

    env = common.TimeLimit(env, config.time_limit)
    env = common.RewardObs(env)
    env = common.ResetObs(env)
    return env

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def play(env, agnt, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics afpss you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    rendered = env.reset()
    rendered = rendered['image']  # use image obs instead
    # rendered = env.render(mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"

    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = config.image_size
    # video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    state = None
    agent_action = None
    MODEL_RENDER = False
    PLAYING_MODE = 'user'
    MODEL_DREAM = False
    MODEL_USE_PRIOR = False

    # todo corrupt state

    action_space = env.action_space['action']
    random_agent = common.RandomAgent(action_space)

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    i = 0

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            if PLAYING_MODE == 'ai' and agent_action is not None:
                hot_action = {'action': agent_action[0].numpy()}

            elif PLAYING_MODE == 'random':
                hot_action, _ = random_agent(obs_input)
                hot_action = {'action': hot_action['action'][0].numpy()}

            else:
                action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                # onehot
                hot_action = np.zeros(env.unwrapped.action_space.n, dtype=np.float32)
                hot_action[action] = 1.0

                hot_action = {'action': hot_action}

            # prev_obs = obs

            obs, rew, env_done, info = env.step(hot_action)


            # i += 1
            # if i == 65:
            #     print('reset')
            #     state = None
            #     i = 0


            # flip image
            # obs['image'] = np.fliplr(obs['image']).copy()
            # obs['image'] = (obs['image'][:,:,::-1]).copy() #color

            # obs['image'] = np.abs(obs['image']-50)

            if len(obs['image'].shape)==1:
                #flatten
                t_obs = torch.from_numpy(obs['image'])[None]
            else:
                t_obs = torch.from_numpy(obs['image'])[None].permute(0, 3, 1, 2)

            obs_input = {'image': t_obs,
                         'action': torch.from_numpy(hot_action['action'][None]),
                         'reward': torch.tensor([rew]), 'reset': torch.tensor([env_done])}  # , 'discount': 1, }

            common.dict_to_device(obs_input, device)

            ########################3
            # def policy(self, obs_input, state=None, mode='train'):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):

                    if state is None:
                        latent = agnt.wm.rssm.initial(len(obs_input['image']), obs_input['image'].device)
                        action = torch.zeros(len(obs_input['image']), agnt._num_act).to(obs_input['image'].device)
                        state = latent, action

                        # random noise
                        latent['logit'] = torch.randn_like(latent['logit'])
                        latent['deter'] = torch.randn_like(latent['deter']) * 0.1

                    # elif obs_input['reset'].any(): # No need for env auto reset for now
                    #     # state = tf.nest.map_structure(lambda x: x * common.pad_dims(#FIXME
                    #     #     1.0 - tf.cast(obs_input['reset'], x.dtype), len(x.shape)), state)
                    #
                    #     # FIXME pls be this
                    #     # TODO So we are reseting (putting to 0) all elements in state, acording to obs_input reset
                    #     latent, action = state
                    #     latent = {k: v * common.pad_dims(1.0 - obs_input['reset'], len(v.shape)) for k, v in latent.items()}
                    #     action = action * common.pad_dims(1.0 - obs_input['reset'], len(action.shape))
                    #     state = latent, action

                    latent, _ = state

                    action = obs_input['action']

                    embed = agnt.wm.encoder(agnt.wm.preprocess(obs_input))
                    sample = not agnt.config.eval_state_mean  # is True

                    if MODEL_DREAM:  # no input information
                        latent = agnt.wm.rssm.img_step(latent, action, sample)
                        recon_latent = latent
                    else:
                        post, prior = agnt.wm.rssm.obs_step(latent, action, embed, sample)
                        latent = post

                        if MODEL_USE_PRIOR:
                            recon_latent = prior
                        else:
                            recon_latent = post

                    # reconstruction
                    feat = agnt.wm.rssm.get_feat(recon_latent)
                    recon = agnt.wm.heads['image'](feat).mode

                    # get actor action
                    actor = agnt._task_behavior.actor(feat)
                    agent_action = actor.mode.cpu()

                    # reward
                    agent_reward = agnt.wm.heads['reward'](feat).mode.item()


                    # if not np.isclose(agent_reward, 0, atol=1e-2):
                    #     print(agent_reward)

                    ### future ideas
                    # elif self._should_expl(self.step):
                    #     actor = self._expl_behavior.actor(feat)
                    #     action = actor.sample()
                    # else:
                    #     actor = self._task_behavior.actor(feat)
                    #     action = actor.sample()
                    # noise = {'train': agnt.config.expl_noise, 'eval': agnt.config.eval_noise}
                    # action = common.action_noise(action, noise[mode], agnt._action_space)

                    state = (latent, None)
                    if len(recon.shape)==4: #image input
                        recon = recon[0].cpu().permute(1, 2, 0).numpy()
                    else:
                        #Flatten
                        image_shape = config.image_size +(1 if config.grayscale else 3,)
                        recon = recon[0].cpu().reshape(image_shape).numpy()

        ################3

        # if callback is not None: #FIXME action
        #     callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            # rendered = env.render(mode='rgb_array')

            if MODEL_RENDER:
                rendered = recon
            else:
                rendered = obs['image']



            if len(rendered.shape)==1:
                #flatted
                rendered = rendered.reshape(1 if config.grayscale else 3,*config.image_size,).transpose(1,2,0)

            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # clear screen
        for x in range(75):
            print('*' * (75 - x), x, end='\x1b[1K\r')

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
                elif event.key == pygame.K_m:
                    if MODEL_RENDER:
                        MODEL_RENDER = False
                    else:
                        MODEL_RENDER = True
                elif event.key == pygame.K_k:
                    if PLAYING_MODE == 'ai':
                        PLAYING_MODE = 'user'
                    else:
                        PLAYING_MODE = 'ai'
                elif event.key == pygame.K_l:
                    if PLAYING_MODE == 'random':
                        PLAYING_MODE = 'user'
                    else:
                        PLAYING_MODE = 'random'
                elif event.key == pygame.K_i:
                    if MODEL_DREAM:
                        MODEL_DREAM = False
                    else:
                        MODEL_DREAM = True
                elif event.key == pygame.K_r:
                    print("reset state")
                    state = None
                elif event.key == pygame.K_f:
                    print("Full reset")
                    state = None
                    env_done = False
                    obs = env.reset()
                # elif event.key == pygame.K_p:
                #     if MODEL_USE_PRIOR:
                #         MODEL_USE_PRIOR = False
                #     else:
                #         MODEL_USE_PRIOR = True



            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        print(f"{color.BOLD}render{color.END}: {'model' if MODEL_RENDER else 'env'}, "
              f"{color.BOLD}player{color.END}: {PLAYING_MODE}, "
              f"{color.BOLD}model{color.END}: {'dreaming' if MODEL_DREAM else 'real game'}", end="")

        # if not MODEL_DREAM:
        #     print(f", {color.BOLD}viewing model{color.END}: {'prior' if MODEL_USE_PRIOR else 'posterior'}",end="")

        print(end="\r")

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)


# def main():
# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4', help='Define Environment')
# args = parser.parse_args()

device = "cpu"

print('Logdir', logdir)

env = make_env("eval")

print('creating agent')
action_space = env.action_space['action']
step = 0

agnt = agent.Agent(config, None, action_space, step)

print("loading agent")
agnt.load_state_dict(torch.load(logdir / 'variables.pt', map_location=device))


agnt.to(device)


print(""" USAGE keys:
r - reset state
f - full reset - reset state + env
m - change view between model / env
k - change to ai playing / user
l - change to random agent playing / user
i - change to dream/imagination mode

""")
# p - view prior/posterior when env input is present # No diff
zoom_ratio = 64/config.image_size[0]
play(env, agnt, zoom=8*zoom_ratio, fps=60/4)


# if __name__ == '__main__':
#     main()
