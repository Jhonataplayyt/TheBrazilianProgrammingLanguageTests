import sys
import os
import subprocess
from pathlib import Path

c = 0
try:
    del sys.argv[0]

    for arg in sys.argv:
        if arg == "-i":
            os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\interpreter\\br-i.exe ' + str(Path(sys.argv[c + 1]).resolve()))

            break
        elif arg == "-c":
            os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\compiler\\br-c.exe ' + str(Path(sys.argv[c + 1]).resolve()))

            break
        elif arg == "-t":
            os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\terminal\\br-t.exe')

            break
        elif arg == "-bc":
            os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\byteCodeCompiler\\br-bc.exe ' + str(Path(sys.argv[c + 1]).resolve()))

            break
        elif arg == "-rbc":
            os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\runByteCodeCompiler\\br-rbc.exe ' + str(Path(sys.argv[c + 1]).resolve()))

            break
        else:
            if os.path.exists(arg):
                os.system(r'"C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\interpreter\\br-i.exe" ' + str(Path(sys.argv[c]).resolve()))
            else:
                print(f"Error: file or parameter not exists '{arg}'")

    c += 1
except:
    os.system(r'C:\\Users\\tempe\\Desktop\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\bin\\terminal\\br-t.exe')