import sys
from functools import wraps

def fast_memorize(func):
    cache: dict = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key: str = str(args) + str(kwargs)

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper