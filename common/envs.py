'''
Code from original Dreamerv2
'''
import os
import threading

import gym
import numpy as np

from functools import reduce

try:
    import cv2
except ImportError:
    cv2 = None


## all environment obs are dicionaries -> image,ram keys

class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        action = action['action']
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky_actions=True, all_actions=False, seed=None):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='rgb', frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions)
        env.seed(seed)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        self._env = env
        self._grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': self._env.observation_space,
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({'action': self._env.action_space})

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.unwrapped.ale.getRAM()}
        return obs

    def step(self, action):
        action = action['action']
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.unwrapped.ale.getRAM()}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


# FIXME finish retro
# WATCH out for multi envinroment
class Retro:

    def __init__(self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30, life_done=False,
                 sticky_actions=True, all_actions=False, seed=None):
        assert cv2 is not None, \
            "opencv-python package not installed!"

        import retro
        env = retro.make(name, use_restricted_actions=retro.Actions.DISCRETE)

        # noops, action_reapeat, resize, end when life is done, image to grayscale
        # env = gym.wrappers.AtariPreprocessing(
        #     env, noops, action_repeat, size[0], life_done, grayscale)
        env.seed(seed)

        env = NoopResetEnv(env, noop_max=noops)
        env = MaxAndSkipEnv(env, skip=action_repeat)

        env = WarpFrame(env, width=size[0], height=size[1], grayscale=grayscale)

        self._env = env
        self._grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': self._env.observation_space,
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({'action': self._env.action_space})

    def close(self):
        self._env.close()

    def reset(self):
        image = self._env.reset()
        obs = {'image': image, 'ram': self._env.unwrapped.get_ram()}
        return obs

    def step(self, action):
        action = action['action']
        image, reward, done, info = self._env.step(action)
        obs = {'image': image, 'ram': self._env.unwrapped.get_ram()}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


class Minigrid:

    def __init__(self, name, size=(84, 84), grayscale=False, seed=None, mode='compact'):
        assert cv2 is not None, \
            "opencv-python package not installed!"

        import gym_minigrid
        env = gym.make(name)

        env.seed(seed)

        if mode == 'compact' or mode == 'compact_full':

            if mode == 'compact_full':
                env = gym_minigrid.wrappers.FullyObsWrapper(env)


            env = MinigridCompactMultiplier(env)


        else:
            env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=9)  # for image only 64 is good (9*7=63)
            env = WarpFrame(env, width=size[0], height=size[1], grayscale=grayscale, dict_space_key='image')

        # noops, action_reapeat, resize, end when life is done, image to grayscale
        # env = gym.wrappers.AtariPreprocessing(
        #     env, noops, action_repeat, size[0], life_done, grayscale)

        # env = NoopResetEnv(env, noop_max=noops)
        # env = MaxAndSkipEnv(env, skip=action_repeat)
        #
        # env = WarpFrame(env,width=size[0], height=size[1], grayscale=grayscale)
        self._env = env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return gym.spaces.Dict({'action': self._env.action_space})

    def close(self):
        self._env.close()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        action = action['action']
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


# Modified only to Retro
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meaning(0) == []

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class Dummy:

    def __init__(self):
        pass

    @property
    def observation_space(self):
        image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        return gym.spaces.Dict({'image': image})

    @property
    def action_space(self):
        action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        obs = {'image': np.zeros((64, 64, 3))}
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = {'image': np.zeros((64, 64, 3))}
        return obs


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env._env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.action_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env._env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:

    def __init__(self, env, key='action'):
        assert isinstance(env.action_space[key], gym.spaces.Discrete)
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env._env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env, key='reward'):
        assert key not in env.observation_space.spaces
        self._env = env
        self._key = key
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env._env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
        return gym.spaces.Dict({
            **self._env.observation_space.spaces, self._key: space})

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class ResetObs:

    def __init__(self, env, key='reset'):
        assert key not in env.observation_space.spaces
        self._env = env
        self._key = key
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env._env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = gym.spaces.Box(0, 1, (), dtype=np.bool)
        return gym.spaces.Dict({
            **self._env.observation_space.spaces, self._key: space})

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reset'] = np.array(False, np.bool)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reset'] = np.array(True, np.bool)
        return obs


class MinigridCompactMultiplier(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs['image'] = obs['image'] * 25
        return obs


#
class FlattenImageObs(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.flatten_space(env.observation_space),
        })

    def observation(self, obs):
        return {'image': obs['image'].transpose(2, 0, 1).flatten()}
