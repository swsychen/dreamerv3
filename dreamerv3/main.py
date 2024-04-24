import importlib
import os
import pathlib
import sys
import warnings
from functools import partial as bind

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
sys.path.insert(0, str(directory.parent.parent))
__package__ = directory.name

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import embodied
from embodied import wrappers


def main(argv=None):

  embodied.print(r"---  ___                           __   ______ ---")
  embodied.print(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---")
  embodied.print(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---")
  embodied.print(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---")

  from . import agent as agt

  # 1.1-- Parse and update env/task argument first, other arguments are updated later (stored in 'other')
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)

  # 1.2-- Load default config (has all needed general+model+default_env hyperparams) from `agt.Agent.configs`, which also has all other env configs.
  config = embodied.Config(agt.Agent.configs['defaults'])

  # 1.3-- Update user-assigned env/task with their properties to the config
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  
  # 1.4-- Update other arguments to the config
  config = embodied.Flags(config).parse(other)
  config = config.update(
      logdir=config.logdir.format(timestamp=embodied.timestamp()))
  
  # 1.5-- Extract a runtime config from the config
  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context)
  print('Run script:', args.script)
  print('Logdir:', args.logdir)

  # 1.6-- Create a LocalPath/GPath object from the logdir for managing path-related operations
  logdir = embodied.Path(args.logdir)
  if args.script not in ('env', 'replay'):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  # TODO: timer and multiprocessing
  def init():
    embodied.timer.global_timer.enabled = args.timer
  embodied.distr.Process.initializers.append(init)
  init()

  if args.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', is_eval=True),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_holdout':
    assert config.eval_dir
    embodied.run.train_holdout(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, config.eval_dir),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', rate_limit=True),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel_env':
    envid = args.env_replica
    if envid < 0:
      envid = int(os.environ['JOB_COMPLETION_INDEX'])
    embodied.run.parallel.env(
        bind(make_env, config), envid, args, False)

  elif args.script == 'parallel_replay':
      embodied.run.parallel.replay(
          bind(make_replay, config, 'replay', rate_limit=True), args)

  else:
    raise NotImplementedError(args.script)


def make_agent(config):
  """Create an agent from the config.

  Args:
      config (Config dict): the all configuation dictionary

  Returns:
      Agent/RandomAgent obj: the agent object
  """
  from . import agent as agt
  env = make_env(config, 0)
  if config.random_agent:
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
  else:
    agent = agt.Agent(env.obs_space, env.act_space, config)
  env.close()
  return agent


def make_logger(config):
  step = embodied.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter, 'Agent'),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(
          logdir, config.run.log_video_fps, config.tensorboard_videos),
      # embodied.logger.WandbOutput(logdir.name, ...),
  ], multiplier)
  return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
  directory = directory and embodied.Path(config.logdir) / directory
  size = int(config.replay.size / 10 if is_eval else config.replay.size)
  length = config.batch_length
  kwargs = {}
  kwargs['online'] = config.replay.online
  if rate_limit and config.run.train_ratio > 0:
    kwargs['samples_per_insert'] = config.run.train_ratio / (
        length - config.replay_context)
    kwargs['tolerance'] = 5 * config.batch_size
    kwargs['min_size'] = min(
        max(config.batch_size, config.run.train_fill), size)
  selectors = embodied.replay.selectors
  if config.replay.fracs.uniform < 1 and not is_eval:
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    import numpy as np
    recency = 1.0 / np.arange(1, size + 1) ** config.replay.recexp
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)
  kwargs['chunksize'] = config.replay.chunksize
  replay = embodied.replay.Replay(length, size, directory, **kwargs)
  return replay


def make_env(config, index, **overrides):
  """Create an environment from the config.

  Args:
      config (Config dict): the all configuation dictionary
      index (_type_): _description_

  Returns:
      _type_: _description_
  """
  suite, task = config.task.split('_', 1)   # the config.task need to be in the format '{suite}_{task}'
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',       # It means that in the module at embodied.envs.dummy, there is a class (or a function) named Dummy
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-ExtraObs-v0', **kw),
  }[suite]                          # extract the env constructor from the suite
  if isinstance(ctor, str):          # typical usage of ctor "module:classname"
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)     # get the env class, here cannout use module.cls directly, because cls is a string, same for the previous line
  kwargs = config.env.get(suite, {})   # get the specific env config from the all config
  kwargs.update(overrides)        # update the env config with the overrides if exists
  if kwargs.pop('use_seed', False):         # pop the key. If not exists, return False
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)      #  generate a deterministic, yet unique, seed value based on the combination of a global seed and an index. Also reproducible for the same global seed and index
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = embodied.Path(config.logdir) / f'env{index}'   # append the index to the end of logdir path---> a new sub logdir for the env index
  env = ctor(task, **kwargs)
  return wrap_env(env, config)

# TODO: need to figure out how the multi-threading works for the environment (atari and minecraft)

def wrap_env(env, config):
  args = config.wrapper        # the wrapper dict will also be output as a Config object (dict)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif not space.discrete:
      env = wrappers.NormalizeAction(env, name)
      if args.discretize:
        env = wrappers.DiscretizeAction(env, name, args.discretize)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
