from Brazilian.Libs.faster import *
from . import utils
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