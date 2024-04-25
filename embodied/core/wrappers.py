import functools
import time

import numpy as np

from . import base
from . import space as spacelib


class TimeLimit(base.Wrapper):

  def __init__(self, env, duration, reset=True):
    """A wrapper for limiting the duration of an episode, mostly for debug, setting a short duration to test the agent

    Args:
        env (obj): the environment to be wrapped
        duration (float,optional): the duration of the episode
        reset (bool, optional): whether to trigger env reset after the timelimit. Defaults to True.
    """
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)      # reset the environment
        return self.env.step(action)
      else:
        action.update(reset=False)    # continue the episode
        obs = self.env.step(action)
        obs['is_first'] = True     # but mark current obs as the first step
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True       # mark the current obs as the last step of the episode when the duration is reached
    self._done = obs['is_last']   # update the exceeding flag 
    return obs


class ActionRepeat(base.Wrapper):

  def __init__(self, env, repeat):
    """Env wrapper to repeat the action for multiple steps. Cannot be used when env itself has explicit action repeat code, otherwsie may cause duplicate repeat action.

    Args:
        env (obj): the environment to be wrapped
        repeat (int): the number of steps to repeat the action
    """
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:   # no need to repeat the action for reset
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):    # repeat the action for multiple steps and accumulate the reward
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs


class ClipAction(base.Wrapper):

  def __init__(self, env, key='action', low=-1, high=1):
    """action clipping wrapper

    Args:
        env (obj): environment to be wrapped
        key (str, optional): the key in action dict that needs to be clipped. Defaults to 'action'.
        low (int, optional): clipping lower bound. Defaults to -1.
        high (int, optional): clipping higher bound. Defaults to 1.
    """
    super().__init__(env)
    self._key = key
    self._low = low
    self._high = high

  def step(self, action):
    clipped = np.clip(action[self._key], self._low, self._high)
    return self.env.step({**action, self._key: clipped})  # update the action value in the action dict (if self._key is 'action', it will overwrite the original action value; otherwise, it will add a new key-value pair)


class NormalizeAction(base.Wrapper):

  def __init__(self, env, key='action'):
    """wrapper for normalized action, it assume normalized action (in finite action space case) is within [-1, 1], 
    will scale the action to the original space for input to the environment. 
    For infinite bound case, normalized and unnormalized action will be the same value. Normalized/unnormlized act_space in this case are both defined to [-1,1].

    Args:
        env (obj): the environment to be wrapped
        key (str, optional): the action key. Defaults to 'action'.
    """
    super().__init__(env)
    self._key = key
    self._space = env.act_space[key]
    self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high) # check if the space bounds are finite
    # the original space bounds 
    self._low = np.where(self._mask, self._space.low, -1)         # use the space bounds if they are finite, otherwise use -1 / 1
    self._high = np.where(self._mask, self._space.high, 1)

  @functools.cached_property
  def act_space(self):
    # set the low and high bounds of the normalized action space to [-1, 1] regardless of whether finite, 
    # because if infinite, the self._mask will all false while self._low and self._high will already be -1 / 1 during init
    low = np.where(self._mask, -np.ones_like(self._low), self._low)  
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = spacelib.Space(np.float32, self._space.shape, low, high)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    # rescale the normalized action value to the original space for input to the environment
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self.env.step({**action, self._key: orig})  # update the action value with original (unnormalized) value in the action dict


class ExpandScalars(base.Wrapper):

  def __init__(self, env):
    """Wrapper of env, to expand the scalar + continuous spaces to have a shape of (1,)

    Args:
        env (obj): the environment to be wrapped
    """
    super().__init__(env)
    self._obs_expanded = []    # record which keys are expanded
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      # expand the scalar + continuous spaces to have a shape of (1,)
      if space.shape == () and key != 'reward' and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._obs_expanded.append(key)
      self._obs_space[key] = space
    self._act_expanded = []
    self._act_space = {}
    for key, space in self.env.act_space.items():
      # expand the scalar + continuous spaces to have a shape of (1,)
      if space.shape == () and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._act_expanded.append(key)
      self._act_space[key] = space

  @functools.cached_property  # it will only compute the first time it is called, and then cache the result
  def obs_space(self):
    return self._obs_space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    """TODO: unknown the purpose of np.squeeze and np.expand_dims

    Args:
        action (_type_): _description_

    Returns:
        _type_: _description_
    """
    action = {
        key: np.squeeze(value, 0) if key in self._act_expanded else value  
        for key, value in action.items()}
    obs = self.env.step(action)
    obs = {
        key: np.expand_dims(value, 0) if key in self._obs_expanded else value
        for key, value in obs.items()}
    return obs


class FlattenTwoDimObs(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._keys = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._keys.append(key)
      self._obs_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  def step(self, action):
    obs = self.env.step(action).copy()
    for key in self._keys:
      obs[key] = obs[key].flatten()
    return obs


class FlattenTwoDimActions(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._origs = {}
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._origs[key] = space.shape
      self._act_space[key] = space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = action.copy()
    for key, shape in self._origs.items():
      action[key] = action[key].reshape(shape)
    return self.env.step(action)


class ForceDtypes(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._obs_space, _, self._obs_outer = self._convert(env.obs_space)
    self._act_space, self._act_inner, _ = self._convert(env.act_space)

  @property
  def obs_space(self):
    return self._obs_space

  @property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = action.copy()
    for key, dtype in self._act_inner.items():
      action[key] = np.asarray(action[key], dtype)
    obs = self.env.step(action)
    for key, dtype in self._obs_outer.items():
      obs[key] = np.asarray(obs[key], dtype)
    return obs

  def _convert(self, spaces):
    results, befores, afters = {}, {}, {}
    for key, space in spaces.items():
      before = after = space.dtype
      if np.issubdtype(before, np.floating):
        after = np.float32
      elif np.issubdtype(before, np.uint8):
        after = np.uint8
      elif np.issubdtype(before, np.integer):
        after = np.int32
      befores[key] = before
      afters[key] = after
      results[key] = spacelib.Space(after, space.shape, space.low, space.high)
    return results, befores, afters


class CheckSpaces(base.Wrapper):

  def __init__(self, env):
    """wrapper for checking the value of the observation and action is of valid type and within the space

    Args:
        env (obj): the environment to be wrapped
    """
    super().__init__(env)

  def step(self, action):
    for key, value in action.items():
      self._check(value, self.env.act_space[key], key)
    obs = self.env.step(action)
    for key, value in obs.items():
      self._check(value, self.env.obs_space[key], key)
    return obs

  def _check(self, value, space, key):
    if not isinstance(value, (
        np.ndarray, np.generic, list, tuple, int, float, bool)):
      raise TypeError(f'Invalid type {type(value)} for key {key}.')
    if value in space:
      return
    dtype = np.array(value).dtype
    shape = np.array(value).shape
    lowest, highest = np.min(value), np.max(value)
    raise ValueError(
        f"Value for '{key}' with dtype {dtype}, shape {shape}, "
        f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(base.Wrapper):

  def __init__(self, env, key='action', bins=5):
    super().__init__(env)
    self._dims = np.squeeze(env.act_space[key].shape, 0).item()
    self._values = np.linspace(-1, 1, bins)
    self._key = key

  @functools.cached_property
  def act_space(self):
    space = spacelib.Space(np.int32, self._dims, 0, len(self._values))
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    continuous = np.take(self._values, action[self._key])
    return self.env.step({**action, self._key: continuous})


class ResizeImage(base.Wrapper):

  def __init__(self, env, size=(64, 64)):
    super().__init__(env)
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = spacelib.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage(base.Wrapper):

  def __init__(self, env, key='image'):
    super().__init__(env)
    self._key = key
    self._shape = self.env.render().shape

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    spaces[self._key] = spacelib.Space(np.uint8, self._shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    obs[self._key] = self.env.render()
    return obs


class BackwardReturn(base.Wrapper):

  def __init__(self, env, horizon):
    super().__init__(env)
    self._discount = 1 - 1 / horizon
    self._bwreturn = 0.0

  @functools.cached_property
  def obs_space(self):
    return {
        **self.env.obs_space,
        'bwreturn': spacelib.Space(np.float32),
    }

  def step(self, action):
    obs = self.env.step(action)
    self._bwreturn *= (1 - obs['is_first']) * self._discount
    self._bwreturn += obs['reward']
    obs['bwreturn'] = np.float32(self._bwreturn)
    return obs


class RestartOnException(base.Wrapper):

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    super().__init__(self._ctor())

  def step(self, action):
    try:
      return self.env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self.env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self.env.step(action)
