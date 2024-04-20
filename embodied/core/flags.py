import re
import sys

from . import config


class Flags:

  def __init__(self, *args, **kwargs):
    """TODO: seems to store the default config
    """
    self._config = config.Config(*args, **kwargs)

  def parse(self, argv=None, help_exits=True):
    parsed, remaining = self.parse_known(argv)
    for flag in remaining:
      if flag.startswith('--') and flag[2:] not in self._config.flat:
        raise KeyError(f"Flag '{flag}' did not match any config keys.")
    if remaining:
      raise ValueError(
          f'Could not parse all arguments. Remaining: {remaining}')
    return parsed

  def parse_known(self, argv=None, help_exits=False):
    """_summary_

    Args:
        argv (list, optional): arguments. Defaults to None.
        help_exits (bool, optional): whether exit after printing help info when querying for help using "--help". Defaults to False.

    Returns:
        tuple: (dict, list) of parsed arguments and remaining arguments (single values without keys specified)
    """
    if argv is None:
      argv = sys.argv[1:]        # read the command line arguments if argv is None
    if '--help' in argv:         # TODO:print the help message (where in the self._config?) and exit if --help is in argv, exit if help_exists is True
      print('\nHelp:')
      lines = str(self._config).split('\n')[2:]
      print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines)) # printing, begin new line with '--' and remove colons, commas, and square brackets from each line x
      help_exits and sys.exit()
    parsed = {}
    remaining = []
    key = None
    vals = None
    for arg in argv:      # iterate through the arguments list (argv), containing "--{key}", "{value}", "--{key}={value}", etc.
      if arg.startswith('--'):
        if key:
          self._submit_entry(key, vals, parsed, remaining)  # submit the existing key and values to the parsed dictionary each time a new key is encountered
        if '=' in arg:
          key, val = arg.split('=', 1)    #split from left, only once
          vals = [val]
        else:
          key, vals = arg, []
      else:
        if key:
          vals.append(arg)
        else:
          remaining.append(arg)   # extract the single value without key specified and add it to the remaining list
    self._submit_entry(key, vals, parsed, remaining)   # submit the last key and values to the parsed dictionary 
    parsed = self._config.update(parsed)
    return parsed, remaining

  def _submit_entry(self, key, vals, parsed, remaining):
    """to check if the key is in the config, and overwrite the default value in the config with the values submitted, then add the key-value pair to the parsed dictionary.
    Unless the key ends with '+', in which case the values will be appended to the default value.

    Args: 
        key (str): the key to be submitted
        vals (list): a list of values to be submitted
        parsed (dict): parsed arguments dictionary
        remaining (list): remaining arguments

    Raises:
        TypeError: if the default value of key in self._config is not a tuple
    """
    if not key and not vals:
      return
    if not key:     # if key is None, then vals should be a single value without a key specified,this case should not happen because it will in this function only if key is not None or key and vals are both None
      vals = ', '.join(f"'{x}'" for x in vals)
      remaining.extend(vals)   # add the values string to the remaining list
      return
      # raise ValueError(f"Values {vals} were not preceded by any flag.")
    name = key[len('--'):]   # remove the '--' from the key to get the name
    if '=' in name:         # handle anomalous cases when '=' is present in the key (should not happen)
      remaining.extend([key] + vals)
      return
    if not vals:        # handle anomalous cases when no values are present for the key
      remaining.extend([key])
      return
      # raise ValueError(f"Flag '{key}' was not followed by any values.")
    if name.endswith('+') and name[:-1] in self._config:  # check if name ends with '+' and the key without the '+' is present in the config
      # Here the trailing '+' is to add specified vals to the default val in the config
      key = name[:-1]
      default = self._config[key]
      if not isinstance(default, tuple):     
        raise TypeError(                                   # default must be a tuple, raise error if it is not
            f"Cannot append to key '{key}' which is of type "
            f"'{type(default).__name__}' instead of tuple.")
      if key not in parsed:       # if key is not already in the parsed dictionary, add it with the default
        parsed[key] = default
      parsed[key] += self._parse_flag_value(default, vals, key)
    elif self._config.IS_PATTERN.fullmatch(name):    # check if name has non-alphanumeric characters like '\ or /', here fullmatch has same effect as match
      pattern = re.compile(name)
      keys = [k for k in self._config.flat if pattern.fullmatch(k)]  # get all keys in config that has the same name as the key, should never find match because all keys in config are checked with no non-alphanumeric characters
      if keys:
        for key in keys:
          # Also overwrite 
          parsed[key] = self._parse_flag_value(self._config[key], vals, key)
      else:
        remaining.extend([key] + vals)     # if the key is not present in the config, add the key and values as items to the remaining list
    elif name in self._config:
      # Here it will overwrite the default value in the config
      key = name
      parsed[key] = self._parse_flag_value(self._config[key], vals, key)
    else:
      remaining.extend([key] + vals)      # if key has no match in the config

  def _parse_flag_value(self, default, value, key):
    value = value if isinstance(value, (tuple, list)) else (value,)
    if isinstance(default, (tuple, list)):
      if len(value) == 1 and ',' in value[0]:           # seperately handle the value with commas, treat each substring between commas as a separate value for the same key
        value = value[0].split(',')
      return tuple(self._parse_flag_value(default[0], [x], key) for x in value)
    if len(value) != 1:            # raise error if more than one value is present for a key 
      raise TypeError(
          f"Expected a single value for key '{key}' but got: {value}")
    value = str(value[0])
    if default is None:
      return value
    if isinstance(default, bool):
      try:
        return bool(['False', 'True'].index(value))
      except ValueError:
        message = f"Expected bool but got '{value}' for key '{key}'."
        raise TypeError(message)
    if isinstance(default, int):
      try:
        value = float(value)  # Allow scientific notation for integers.
        assert float(int(value)) == value
      except (ValueError, TypeError, AssertionError):
        message = f"Expected int but got '{value}' for key '{key}'."
        raise TypeError(message)
      return int(value)
    if isinstance(default, dict):
      raise KeyError(
          f"Key '{key}' refers to a whole dict. Please speicfy a subkey.")
    try:
      return type(default)(value)
    except ValueError:
      raise TypeError(
          f"Cannot convert '{value}' to type '{type(default).__name__}' for "
          f"key '{key}'.")
