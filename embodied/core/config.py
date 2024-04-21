import io
import json
import re

from . import path


class Config(dict):

  SEP = '.'
  IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

  def __init__(self, *args, **kwargs):
    """Initializes itself as a standard dictionary with its contents set to those of self._nested, but with additional methods and properties
    Here needs to assign the values to the base class dictionary so that conversion to dict `filename.write(json.dumps(dict(self)))` does not lose the content.
    
    args, kwargs: the input key-value pairs to be stored in the Config dictionary
    """
    mapping = dict(*args, **kwargs)
    mapping = self._flatten(mapping)
    mapping = self._ensure_keys(mapping)
    mapping = self._ensure_values(mapping)
    self._flat = mapping
    self._nested = self._nest(mapping)
    # Need to assign the values to the base class dictionary so that
    # conversion to dict does not lose the content.
    super().__init__(self._nested)

  @property
  def flat(self):
    return self._flat.copy()

  def save(self, filename):
    """Saves the configuration to a file in either JSON or YAML format

    Args:
        filename (str): the path to the file to save the configuration to

    Raises:
        NotImplementedError: if the file extension is not .json, .yml, or .yaml
    """
    filename = path.Path(filename)
    if filename.suffix == '.json':
      filename.write(json.dumps(dict(self)))
    elif filename.suffix in ('.yml', '.yaml'):
      from ruamel.yaml import YAML
      yaml = YAML(typ='safe')
      with io.StringIO() as stream:
        yaml.dump(dict(self), stream)
        filename.write(stream.getvalue())
    else:
      raise NotImplementedError(filename.suffix)

  @classmethod
  def load(cls, filename):
    filename = path.Path(filename)
    if filename.suffix == '.json':
      return cls(json.loads(filename.read_text()))
    elif filename.suffix in ('.yml', '.yaml'):
      from ruamel.yaml import YAML
      yaml = YAML(typ='safe')
      return cls(yaml.load(filename.read_text()))
    else:
      raise NotImplementedError(filename.suffix)

  def __contains__(self, name):
    """overriding the __contains__ method to allow for nested dictionary access using the SEP as a separator

    Usage:
        `'key1.key2.key3' in self `will return True if the key3 is in the nested dictionary of key2 in the nested dictionary of key1

    Args:
        name (str): key that may contain the SEP as a separator

    Returns:
        boolean: True if the key is in the dictionary, False otherwise
    """
    try:
      self[name]
      return True
    except KeyError:
      return False

  def __getattr__(self, name):
    if name.startswith('_'):
      return super().__getattr__(name)
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __getitem__(self, name):
    """overriding the __getitem__ method to allow for nested dictionary access using the SEP as a separator
    
    Usage:
        `self['key1.key2.key3']` will return the value of the key3 in the nested dictionary of key2 in the nested dictionary of key1
    
    Args:
        name (str): key that may contain the SEP as a separator

    Raises:
        KeyError: if the key is not found in the dictionary

    Returns:
        Optional: the value of the key in the dictionary, it will be converted to a Config dict object if it is a dict
    """
    result = self._nested
    for part in name.split(self.SEP):
      try:
        result = result[part]
      except TypeError:
        raise KeyError
    if isinstance(result, dict):
      result = type(self)(result)
    return result

  def __setattr__(self, key, value):
    if key.startswith('_'):
      return super().__setattr__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def __setitem__(self, key, value):
    if key.startswith('_'):
      return super().__setitem__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def __reduce__(self):
    return (type(self), (dict(self),))

  def __str__(self):
    lines = ['\nConfig:']
    keys, vals, typs = [], [], []
    for key, val in self.flat.items():
      keys.append(key + ':')
      vals.append(self._format_value(val))
      typs.append(self._format_type(val))
    max_key = max(len(k) for k in keys) if keys else 0
    max_val = max(len(v) for v in vals) if vals else 0
    for key, val, typ in zip(keys, vals, typs):
      key = key.ljust(max_key)
      val = val.ljust(max_val)
      lines.append(f'{key}  {val}  ({typ})')
    return '\n'.join(lines)

  def update(self, *args, **kwargs):
    """update the config dict with new key-value pairs from the input args and kwargs, here will also check the value type consistency and convert the value to the old value type.
    The key to be updated must exist in the config dict.

    Raises:
        KeyError: if no keys are found in pattern match or is empty, raise error
        ValueError: if the new value is a fractional float and the old value is an int, raise error
        TypeError: if the new value cannot be converted to the old value type, raise error

    Returns:
        Config obj: a new Config object with the updated values and other values unchanged
    """
    result = self._flat.copy()       # make a copy of the flattened config dict
    inputs = self._flatten(dict(*args, **kwargs))    # flatten the input dict
    for key, new in inputs.items():
      if self.IS_PATTERN.match(key):
        pattern = re.compile(key)
        keys = {k for k in result if pattern.match(k)}   #TODO: I think should be fullmatch instead of match, unknown author's intention about the non-alphanumeric characters
      else:
        keys = [key]
      if not keys:        # if no keys are found in pattern match, raise error
        raise KeyError(f'Unknown key or pattern {key}.')
      for key in keys:
        old = result[key]  #if key not found, will get KeyError
        try:              # Here repeat again the type checking and conversion, as in _parse_flag_value()
          if isinstance(old, int) and isinstance(new, float):
            if float(int(new)) != new:
              message = f"Cannot convert fractional float {new} to int."
              raise ValueError(message)
          result[key] = type(old)(new)         # replace the old value with the new value
        except (ValueError, TypeError):
          raise TypeError(
              f"Cannot convert '{new}' to type '{type(old).__name__}' " +
              f"for key '{key}' with previous value '{old}'.")
    return type(self)(result)   # return a new Config object with the updated values

  def _flatten(self, mapping):
    """flatten the nested dictionary into a single level dictionary and change the keys with their relative paths,
    more efficient to lookup, check and modification than the recursive version

    Args:
        mapping (dict): input nested dictionary

    Returns:
        dict: flattened dictionary
    """
    result = {}
    for key, value in mapping.items():
      if isinstance(value, dict):
        for k, v in self._flatten(value).items():
          if self.IS_PATTERN.match(key) or self.IS_PATTERN.match(k):
            combined = f'{key}\\{self.SEP}{k}'
          else:
            combined = f'{key}{self.SEP}{k}'
          result[combined] = v
      else:
        result[key] = value
    return result

  def _nest(self, mapping):
    """re-nest the flattened dictionary into a nested dictionary according to the SEP in the keys

    Args:
        mapping (dict): flattened dictionary

    Returns:
        dict: re-nested dictionary
    """
    result = {}
    for key, value in mapping.items():
      parts = key.split(self.SEP)
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _ensure_keys(self, mapping):
    """ensure that the keys in the (flattened) mapping do not contain any non-alphanumeric characters,
    usually used after self._flatten()

    Args:
        mapping (dict): input dictionary

    Returns:
        dict: input dictionary
    """
    for key in mapping:
      assert not self.IS_PATTERN.match(key), key
    return mapping

  def _ensure_values(self, mapping):
    """ensure that the values in the (flattened) mapping are of the correct type and format.
     all lists will be converted to tuples, and tuples will be checked for type consistency (must be all of the same type from strings, floats, ints, bools)

    Args:
        mapping (dict): input dictionary

    Raises:
        TypeError: no empty lists allowed
        TypeError: lists can only contain strings, floats, ints, bools
        TypeError: elements of a list must all be of the same type

    Returns:
        dict: a deep copy of the input dictionary with lists becoming tuples, the original dictionary is not modified
    """
    result = json.loads(json.dumps(mapping))
    for key, value in result.items():
      if isinstance(value, list):
        value = tuple(value)
      if isinstance(value, tuple):
        if len(value) == 0:
          message = 'Empty lists are disallowed because their type is unclear.'
          raise TypeError(message)
        if not isinstance(value[0], (str, float, int, bool)):
          message = 'Lists can only contain strings, floats, ints, bools'
          message += f' but not {type(value[0])}'
          raise TypeError(message)
        if not all(isinstance(x, type(value[0])) for x in value[1:]):
          message = 'Elements of a list must all be of the same type.'
          raise TypeError(message)
      result[key] = value
    return result

  def _format_value(self, value):
    if isinstance(value, (list, tuple)):
      return '[' + ', '.join(self._format_value(x) for x in value) + ']'
    return str(value)

  def _format_type(self, value):
    if isinstance(value, (list, tuple)):
      assert len(value) > 0, value
      return self._format_type(value[0]) + 's'
    return str(type(value).__name__)
