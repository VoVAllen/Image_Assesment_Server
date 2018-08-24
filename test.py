"""
@debounce(3)
def hi(name):
    print('hi {}'.format(name))
hi('dude')
time.sleep(1)
hi('mike')
time.sleep(1)
hi('mary')
time.sleep(1)
hi('jane')
"""
import time
import asyncio


def debounce(s):
    """Decorator ensures function that can only be called once every `s` seconds.
    """

    def decorate(f):
        t = None
        arg_list = []

        def wrapped(*args, **kwargs):
            nonlocal t
            nonlocal arg_list
            arg_list.append(*args)
            print(arg_list)
            t_ = time.time()
            if t is None or t_ - t >= s:
                result = f(arg_list, **kwargs)
                arg_list = []
                t = time.time()
                return result

        return wrapped

    return decorate


@debounce(3)
def hi(name):
    dict = {
        'dude': 1,
        'mike': 2,
        'mary': 3,
        'jane': 4
    }
    print('hi {}'.format(name))


hi('dude')
time.sleep(1)
hi('mike')
time.sleep(1)
hi('mary')
time.sleep(1)
hi('jane')
