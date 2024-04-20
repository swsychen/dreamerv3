class Singleton:
    _instance = None  # Class attribute to store the singleton instance
    _is_initialized = False  # Class attribute to store the initialization state
    def __new__(cls):
        if cls._instance is None:
            print("Creating the instance")
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._is_initialized:
            print("Instance already initialized")
            return
        # Initialization method might be called multiple times
        # when attempting to create more instances
        print("Initializing the instance")
        self._is_initialized = True

# Usage of the Singleton class
singleton1 = Singleton()
singleton2 = Singleton()

# Check if both variables point to the same instance
print("Singleton1 is singleton2:", singleton1 is singleton2)

# Output additional instance properties or states to show it's the same instance
print("ID of singleton1:", id(singleton1))
print("ID of singleton2:", id(singleton2))


class MyClass:
    shared_var = 10

instance1 = MyClass()
instance2 = MyClass()

print(instance1.shared_var)  # Output: 10
print(instance2.shared_var)  # Output: 10

instance1.shared_var = 20

print(instance1.shared_var)  # Output: 20
print(instance2.shared_var)  # Output: 20

somepath="/home/username/folder/file.txt"
ss = somepath.rsplit('/', 1)
print(ss)

import re
pattern = re.compile("abc")
result = pattern.match("abcdef")
# result is a match object since "abcdef" starts with "abc"
result2 = pattern.match("defabc")
# result2 is None because "defabc" does not start with "abc"
print(result, result2)

pattern = re.compile("abc")
result = pattern.fullmatch("abcdef")
# result is None because the entire string "abcdef" does not conform to "abc"
result2 = pattern.fullmatch("abc")
# result2 is a match object because the entire string "abc" perfectly matches "abc"
print(result, result2)
vals=['a', 'b', 'c']
vals = 'x '.join(f"'{x}'" for x in vals)
print(vals)

sss="abc=1=2=3"
print(sss.split('=', 1))


IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')
name = "abc"
print(IS_PATTERN.fullmatch(name))
