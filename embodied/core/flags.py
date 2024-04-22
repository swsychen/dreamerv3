import re
import sys

from . import config


class Flags:

  def __init__(self, *args, **kwargs):
    """store the config, as a high-level interface to the config.Config class for parsing command line arguments. and updating the config with the parsed arguments.
    """
    self._config = config.Config(*args, **kwargs)

  def parse(self, argv=None, help_exits=True):
    """[process argv ,get new config and error for unknown key] process the command line arguments and get the new config with the parsed arguments.

    Args:
        argv (list, optional): arguments to be submitted. Defaults to None.
        help_exits (bool, optional): <not used in code>, should be used for whether exit after printing help info when querying for help using "--help". Defaults to True.

    Raises:
        KeyError: the key flag did not match any config keys
        ValueError: could not parse all arguments, remaining arguments list is not empty

    Returns:
        Config obj (a custom dict type): updated config dict with the parsed arguments
    """
    parsed, remaining = self.parse_known(argv)
    for flag in remaining:         
      if flag.startswith('--') and flag[2:] not in self._config.flat:  # it it is a key and not in the config, raise error
        raise KeyError(f"Flag '{flag}' did not match any config keys.")
    if remaining:        # if remaining list it not empty, raise error
      raise ValueError(
          f'Could not parse all arguments. Remaining: {remaining}')
    return parsed

  def parse_known(self, argv=None, help_exits=False):
    """[parse argv ,update config, no error for unknown key] parse the command line arguments and update the config with the parsed arguments. If input argv is None, it will read the system command line arguments. 
    If '--help' is in the arguments, it will print the help message and exit if help_exits is True.

    Args:
        argv (list, optional): arguments. Defaults to None.
        help_exits (bool, optional): whether exit after printing help info when querying for help using "--help". Defaults to False.

    Returns:
        tuple: (Config obj, list) of updated config dict and remaining arguments (like single values without keys specified, or non-exist keys in config...)
    """
    if argv is None:
      argv = sys.argv[1:]        # read the command line arguments if argv is None
    if '--help' in argv:         # print the help message (the config dict in rows of string format) and exit if --help is in argv, exit if help_exists is True
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
          key, vals = arg, []         # all vals will be in a list
      else:
        if key:
          vals.append(arg)
        else:
          remaining.append(arg)   # extract the single value without key specified and add it to the remaining list
    self._submit_entry(key, vals, parsed, remaining)   # submit the last key and values to the parsed dictionary 
    parsed = self._config.update(parsed)   # update the config with the parsed dictionary (have new values from command line)
    return parsed, remaining

  def _submit_entry(self, key, vals, parsed, remaining):
    """[check key, add/replace value] to check if the key is in the config, and overwrite the default value in the config with the values submitted, then add the key-value pair to the parsed dictionary.
    Unless the key ends with '+', in which case the values will be appended to the default value. If there is no match of the key in the config/ key or value is None/ have "=" in key, the key (or/and) values will be added to the remaining list.

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
      # Here it will not use/ overwrite the default value in the config
      key = name
      parsed[key] = self._parse_flag_value(self._config[key], vals, key)
    else:
      remaining.extend([key] + vals)      # if key has no match in the config

  def _parse_flag_value(self, default, value, key):
    """[check value, convert value] check if the value represents the same type as the default value for boolean, integer default types. Convert the value to the default value type for these scenarios and other types.
    It can handle intended multiple values for a key, but it needs to be a single-item list like ['value1, value2, ...'], where the values are separated by commas.
    In this function, it doesn't decide whether to replace the default value with the submitted value, but only convert the submitted val from string to the matched type, preparing for the subsequent decision-making or other process.

    Args:
        default (tuple/item): the default value of the key in the config
        value (list): a list of values to be submitted, should be a single-item list like ['value'] or ['value1, value2, ...']
        key (str): the key to be submitted

    Raises:
        TypeError: if more than one item in value list is simultaneously present for a key 
        TypeError: if the value is not 'False' or 'True' for a boolean default value
        TypeError: if the value cannot be converted / is not the int type for an integer default value
        KeyError: if the default value is a dictionary
        TypeError: if the value cannot be converted to the type of the default value

    Returns:
        tuple/item: the converted value, which is the same type as the default value. tuple if it is multiple-values case, single item if it is a single value
    """
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
      return value              # use the submitted value if default is None
    if isinstance(default, bool):
      try:
        return bool(['False', 'True'].index(value)) # to find the index of the string value in the list ['False', 'True'].---> value: 'False'->0, 'True'->1
      except ValueError:                # if value is not 'False' or 'True', raise error
        message = f"Expected bool but got '{value}' for key '{key}'."
        raise TypeError(message)
    if isinstance(default, int):
      try:
        value = float(value)  # Allow scientific notation for integers.
        assert float(int(value)) == value       # make sure the value is an integer if default is an integer
      except (ValueError, TypeError, AssertionError):
        message = f"Expected int but got '{value}' for key '{key}'."
        raise TypeError(message)
      return int(value)
    if isinstance(default, dict):    # if default is a dictionary, raise error
      raise KeyError(
          f"Key '{key}' refers to a whole dict. Please speicfy a subkey.")
    try:
      return type(default)(value)     # convert the value to the type of the default value
    except ValueError:
      raise TypeError(
          f"Cannot convert '{value}' to type '{type(default).__name__}' for "
          f"key '{key}'.")
