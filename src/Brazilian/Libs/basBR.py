from Brazilian.Libs.faster import *
from ctypes import wintypes
from . import utils
import ctypes
import sys
import os

def input_char(msg: str):
    print(msg, end='')

    result = None

    if os.name == 'nt':
        import msvcrt

        result = msvcrt.getch()
    else:
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        ch = None

        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        result = ch
    
    try:
        return result.decode('utf-8')
    except:
        return result


if os.name == "nt":
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # Definições de VirtualAlloc/VirtualFree aqui
    kernel32.VirtualAlloc.restype = wintypes.LPVOID
    kernel32.VirtualAlloc.argtypes = [wintypes.LPVOID,
                                      ctypes.c_size_t,
                                      wintypes.DWORD,
                                      wintypes.DWORD]

    kernel32.VirtualFree.restype = wintypes.BOOL
    kernel32.VirtualFree.argtypes = [wintypes.LPVOID,
                                     ctypes.c_size_t,
                                     wintypes.DWORD]

    def virtualAlloc(lpAddress, dwSize, flAllocationType, flProtect):
        return kernel32.VirtualAlloc(lpAddress, dwSize, flAllocationType, flProtect)

    def virtualFree(lpAddress, dwSize, dwFreeType):
        return kernel32.VirtualFree(lpAddress, dwSize, dwFreeType)

else:
    libc = ctypes.CDLL(None)

    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]

    libc.free.restype = None
    libc.free.argtypes = [ctypes.c_void_p]

    def malloc(x):
        return libc.malloc(x)

    def free(x):
        return libc.free(x)

def memmove(ptr, x, _len):
    return ctypes.memmove(ptr, x, _len)

