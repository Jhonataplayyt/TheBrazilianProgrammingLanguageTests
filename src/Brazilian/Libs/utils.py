import platform
import os

def is_android():
    return (
        "ANDROID_ROOT" in os.environ
        or "ANDROID_DATA" in os.environ
        or "android" in platform.release().lower()
    )

_rt = None

if platform.system() == "Linux":
    _rt = "/"
elif platform.system() == "Windows":
    _rt = "C:/"
elif is_android():
    inp = input('What is the name of the package of the app you are using? : ')

    _rt = os.path.join('/data/data/', inp)
else:
    raise OSError("unsupported os.")

def make_files(rt, mode='w'):
    directory = os.path.dirname(rt)

    os.makedirs(directory, exist_ok=True)

    return open(rt, mode)

def find_abs_path(frac):
    global _rt
    global _dir

    norm_frac = os.path.normpath(frac)

    search_bases = [
        _rt,
    ]

    for base in search_bases:
        for root, dirs, files in os.walk(base):
            for f in files:
                current = os.path.join(root, f)
                if current.endswith(norm_frac):
                    return os.path.abspath(current)

    return None

