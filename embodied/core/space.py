import numpy as np


class Space:

  def __init__(self, dtype, shape=(), low=None, high=None):
    """An object for representing a space of possible values. Usually used for observations and actions.

    Args:
        dtype (type alias, str): value type.
        shape (tuple, optional): shape of the space. Defaults to (), which represents scalar.
        low (int/float/optional): lower bound of the possible values. Defaults to None.
        high (int/float/optional): higher bound of the possible values (excluded). Defaults to None.
    """
    # For integer types, high is one above the highest allowed value.
    shape = (shape,) if isinstance(shape, int) else shape
    self._dtype = np.dtype(dtype)
    assert self._dtype is not object, self._dtype  # basically useless
    assert isinstance(shape, tuple), shape
    # infer the low, high, and shape of the space if some of them are not provided
    self._low = self._infer_low(dtype, shape, low, high)
    self._high = self._infer_high(dtype, shape, low, high)
    self._shape = self._infer_shape(dtype, shape, self._low, self._high)
    self._discrete = (
        np.issubdtype(self.dtype, np.integer) or self.dtype == bool)  # check if the dtype is integer or boolean--> discrete
    self._random = np.random.RandomState()  # a random number generator independent of the global random number generator state
                                            # usage example: random_float = self._random.rand()

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  @property
  def discrete(self):
    return self._discrete

  @property
  def classes(self):
    """calculate the number of classes for a discrete space

    Returns:
        Optional: the number of classes if the space is discrete, otherwise raise an assertion error
    """
    assert self.discrete
    classes = self._high - self._low
    if not classes.ndim:     # if classes is a scalar
      classes = int(classes.item())
    return classes

  def __repr__(self):
    low = None if self.low is None else self.low.min()
    high = None if self.high is None else self.high.min()
    return (
        f'Space(dtype={self.dtype.name}, '
        f'shape={self.shape}, '
        f'low={low}, '
        f'high={high})')

  def __contains__(self, value):
    value = np.asarray(value)
    if np.issubdtype(self.dtype, str):
      return np.issubdtype(value.dtype, str)
    if value.shape != self.shape:
      return False
    if (value > self.high).any():
      return False
    if (value < self.low).any():
      return False
    if value.dtype != self.dtype:
      return False
    return True

  def sample(self):
    low, high = self.low, self.high
    if np.issubdtype(self.dtype, np.floating):
      low = np.maximum(np.ones(self.shape) * np.finfo(self.dtype).min, low)
      high = np.minimum(np.ones(self.shape) * np.finfo(self.dtype).max, high)
    return self._random.uniform(low, high, self.shape).astype(self.dtype)

  def _infer_low(self, dtype, shape, low, high):
    """get the lower bound of the possible values, will use the -inf or possible minimum value of the dtype if not provided

    Args:
        dtype (type alias/str): value type.
        shape (tuple): shape of the space.
        low (int/float/optional): lower bound of the possible values. Defaults to None.
        high (int/float/optional): <unused>

    Raises:
        ValueError: shape broadcasting error
        ValueError: Cannot infer low bound from shape and dtype.

    Returns:
        array: an array of the lower bound of the possible values, shape is given by the input shape
    """
    if np.issubdtype(dtype, str):
      assert low is None, low
      return None            # None for string type
    if low is not None:
      try:
        return np.broadcast_to(low, shape)
      except ValueError:
        raise ValueError(f'Cannot broadcast {low} to shape {shape}')
    elif np.issubdtype(dtype, np.floating):
      return -np.inf * np.ones(shape)        # -inf for floating point types
    elif np.issubdtype(dtype, np.integer):
      return np.iinfo(dtype).min * np.ones(shape, dtype)  # get the minimum value of the integer type
    elif np.issubdtype(dtype, bool):
      return np.zeros(shape, bool)        # 0 for boolean type
    else:
      raise ValueError('Cannot infer low bound from shape and dtype.')

  def _infer_high(self, dtype, shape, low, high):
    if np.issubdtype(dtype, str):
      assert high is None, high
      return None
    if high is not None:
      try:
        return np.broadcast_to(high, shape)
      except ValueError:
        raise ValueError(f'Cannot broadcast {high} to shape {shape}')
    elif np.issubdtype(dtype, np.floating):
      return np.inf * np.ones(shape)
    elif np.issubdtype(dtype, np.integer):
      return np.iinfo(dtype).max * np.ones(shape, dtype)
    elif np.issubdtype(dtype, bool):
      return np.ones(shape, bool)
    else:
      raise ValueError('Cannot infer high bound from shape and dtype.')

  def _infer_shape(self, dtype, shape, low, high):
    """Infer the shape of the space from the space input or the low and high bounds.

    Args:
        dtype (_type_): <unused>
        shape (tuple, optional): the shape of the space
        low (array, optional): the lower bound of the possible values.
        high (array, optional): the higher bound of the possible values.

    Returns:
        tuple: the shape of the space
    """
    if shape is None and low is not None:
      shape = low.shape
    if shape is None and high is not None:
      shape = high.shape
    if not hasattr(shape, '__len__'):
      shape = (shape,)          # convert to vector if shape is a scalar
    assert all(dim and dim > 0 for dim in shape), shape   # shape could be a () empty tuple, and it will pass the check
    return tuple(shape)
