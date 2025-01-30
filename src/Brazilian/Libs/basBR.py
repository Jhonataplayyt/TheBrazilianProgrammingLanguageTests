import ctypes
import platform
from Brazilian.Libs.faster import *

system = platform.system()

LibPath = None

if system == 'Windows':
    LibPath = r'C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\src\\Brazilian\\Base\\basBR.dll'
elif system == 'Linux':
    LibPath = r'C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\src\\Brazilian\\Base\\basBR.so'
elif system == 'Darwin':
    LibPath = r'C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\src\\Brazilian\\Base\\basBR.dylib'
else:
    raise OSError('OS not supported.')

basBR = ctypes.CDLL(LibPath, winmode=0)

basBR.input_char.argtypes = [ctypes.c_char_p]
basBR.input_char.restype = ctypes.c_char_p

def input_char(msg: str):
    msg_bytes = msg.encode('utf-8')

    result = basBR.input_char(msg_bytes)

    result_str = result

    try:
        return result_str.decode('utf-8')
    except:
        return result_str