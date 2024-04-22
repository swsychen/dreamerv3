import contextlib
import glob as globlib
import os
import re
import shutil


class Path:

  # Use __slots__ to statically define memory structure of class instances.
  # This restricts instance attributes to only '_path', saving memory by avoiding the creation of __dict__ for each instance.
  # This results in faster attribute access and reduced memory use, but limits ability to add new attributes dynamically and use weak references unless explicitly handled.
  __slots__ = ('_path',)      

  filesystems = []

  def __new__(cls, path):
    """__new__ is responsible for actually creating the instance, before __init__ is called. It's a class method that returns a new instance of the class cls. 
    The first argument is the class itself (cls), and the rest of the arguments are the same as the arguments to the __init__ method. 
    __new__ is used when you need to control the creation of a new instance. One common use of __new__ is to create a singleton class, where only one instance of the class is ever created.

    Args:
        path (str): the path to be stored in the Path object

    Raises:
        NotImplementedError: if path not supported by any filesystem (local or Google cloud storage)

    Returns:
        obj: a new instance of the class cls
    """
    if cls is not Path:           # if the class cls is a subclass of Path,
      return super().__new__(cls) # create a new instance of the class cls.
    path = str(path)
    for impl, pred in cls.filesystems:   # iterate over the filesystems (GFilePath, LocalPath) and their predicates (lambda path: path.startswith('gs://'), lambda path: True
      if pred(path):                      # if the path is supported, return a new instance of the class impl (GFilePath or LocalPath)
        obj = super().__new__(impl)
        obj.__init__(path)    # Here it is unnecessary to call __init__ explicitly, as it is called automatically after __new__ returns the new instance. This line will make the initialization happen twice.
        return obj 
    raise NotImplementedError(f'No filesystem supports: {path}')

  def __getnewargs__(self):
    return (self._path,)

  def __init__(self, path):
    """initializes the Path object, containing methods for path management, the input path will be modified to remove leading dots or leading dot slashes and single trailing slash.
    And empty path will be represented by a dot.
    
    Args:
        path (str): the path to be stored in the Path object
    """
    assert isinstance(path, str)
    path = re.sub(r'^\./*', '', path)  # Remove leading dot or dot slashes.
    path = re.sub(r'(?<=[^/])/$', '', path)  # Remove single trailing slash.
    path = path or '.'  # Empty path is represented by a dot.
    self._path = path

  def __truediv__(self, part):
    """overloads the division operator to concatenate the path with the input part, and returns a new instance of the class with the new path.
    It will automatically decide whether to add a slash "/" between them.

    Args:
        part (str): the part to be concatenated to the end of the path

    Returns:
        Path: a new instance of the class with the concatenated path
    """
    sep = '' if self._path.endswith('/') else '/'
    return type(self)(f'{self._path}{sep}{str(part)}')

  def __repr__(self):
    return f'Path({str(self)})'

  def __fspath__(self):
    return str(self)

  def __eq__(self, other):
    return self._path == other._path

  def __lt__(self, other):
    return self._path < other._path

  def __str__(self):
    """returns the path stored in the Path object if called like str(Path)

    Returns:
        str: stored path
    """
    return self._path

  @property
  def parent(self):
    """returns the parent directory of the path, if the path does not contain a slash, it returns the current directory "."
     if the path is a root directory, it returns the root directory "/"

    Returns:
        obj: a new instance of the class with the parent directory as the path input
    """
    if '/' not in self._path:   
      return type(self)('.')
    parent = self._path.rsplit('/', 1)[0]   # splitting beginning from the right, and do only one split, effectively returning the parent directory with [0]
    parent = parent or ('/' if self._path.startswith('/') else '.')
    return type(self)(parent)

  @property
  def name(self):
    if '/' not in self._path:
      return self._path
    return self._path.rsplit('/', 1)[1]

  @property
  def stem(self):
    return self.name.split('.', 1)[0] if '.' in self.name else self.name

  @property
  def suffix(self):
    return ('.' + self.name.split('.', 1)[1]) if '.' in self.name else ''

  def read(self, mode='r'):
    assert mode in 'r rb'.split(), mode
    with self.open(mode) as f:
      return f.read()

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    with self.open(mode) as f:
      f.write(content)

  @contextlib.contextmanager
  def open(self, mode='r'):
    raise NotImplementedError

  def absolute(self):
    raise NotImplementedError

  def glob(self, pattern):
    raise NotImplementedError

  def exists(self):
    raise NotImplementedError

  def isfile(self):
    raise NotImplementedError

  def isdir(self):
    raise NotImplementedError

  def mkdir(self):
    raise NotImplementedError

  def remove(self):
    raise NotImplementedError

  def rmtree(self):
    raise NotImplementedError

  def copy(self, dest):
    raise NotImplementedError

  def move(self, dest):
    self.copy(dest)
    self.remove()


class LocalPath(Path):

  __slots__ = ('_path',)

  def __init__(self, path):
    super().__init__(os.path.expanduser(str(path)))   #expand the user's home directory in the path, from ~ to /home/username

  @contextlib.contextmanager
  def open(self, mode='r'):
    """opens the file in the path in read mode, and yields the file handler. 
    This file handler to the outer block is temporary and meantime the inner block will be suspended at yield, and will continue after the outer block is finished.
    Here after the outer block is finished, the file handler will be closed automatically because the inner 'with' block is exited. 

    Usage:
        with path.open('r') as f:
            content = f.read()

    Args:
        mode (str, optional): _description_. Defaults to 'r'.

    Yields:
        FileHandler: a file handler for further operations
    """
    with open(str(self), mode=mode) as f:
      yield f

  def absolute(self):
    return type(self)(os.path.absolute(str(self)))

  def glob(self, pattern):
    for path in globlib.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return os.path.exists(str(self))

  def isfile(self):
    return os.path.isfile(str(self))

  def isdir(self):
    return os.path.isdir(str(self))

  def mkdir(self):
    os.makedirs(str(self), exist_ok=True)

  def remove(self):
    os.rmdir(str(self)) if self.isdir() else os.remove(str(self))

  def rmtree(self):
    shutil.rmtree(self)

  def copy(self, dest):
    if self.isfile():
      shutil.copy(self, type(self)(dest))
    else:
      shutil.copytree(self, type(self)(dest), dirs_exist_ok=True)

  def move(self, dest):
    shutil.move(self, dest)


class GFilePath(Path):

  __slots__ = ('_path',)

  gfile = None

  def __init__(self, path):
    path = str(path)
    if not (path.startswith('/') or '://' in path):
      path = os.path.abspath(os.path.expanduser(path))
    super().__init__(path)
    if not type(self).gfile:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
      type(self).gfile = tf.io.gfile

  @contextlib.contextmanager
  def open(self, mode='r'):
    path = str(self)
    if 'a' in mode and path.startswith('/cns/'):
      path += '%r=3.2'
    if mode.startswith('x') and self.exists():
      raise FileExistsError(path)
      mode = mode.replace('x', 'w')
    with self.gfile.GFile(path, mode) as f:
      yield f

  def absolute(self):
    return self

  def glob(self, pattern):
    for path in self.gfile.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return self.gfile.exists(str(self))

  def isfile(self):
    return self.exists() and not self.isdir()

  def isdir(self):
    return self.gfile.isdir(str(self))

  def mkdir(self):
    self.gfile.makedirs(str(self))

  def remove(self):
    self.gfile.remove(str(self))

  def rmtree(self):
    self.gfile.rmtree(str(self))

  def copy(self, dest):
    dest = type(self)(dest)
    if self.isfile():
      self.gfile.copy(str(self), str(dest), overwrite=True)
    else:
      for folder, subdirs, files in self.gfile.walk(str(self)):
        target = type(self)(folder.replace(str(self), str(dest)))
        target.exists() or target.mkdir()
        for file in files:
          (type(self)(folder) / file).copy(target / file)

  def move(self, dest):
    dest = Path(dest)
    if dest.isdir():
      dest.rmtree()
    self.gfile.rename(self, str(dest), overwrite=True)


Path.filesystems = [
    (GFilePath, lambda path: path.startswith('gs://')),
    (LocalPath, lambda path: True),
]
