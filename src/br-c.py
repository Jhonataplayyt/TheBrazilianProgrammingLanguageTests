from Brazilian import *
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else ""
    if not fn:
        print("Error: We need a file for the compiler to work.", file=sys.stderr)
        return

    code = open(isLangExtension(fn), 'r', encoding='utf-8').read()

    cwd = Path.cwd()
    build_dir = cwd / 'build'
    temp_dir = cwd / 'temp'
    brazilian_file = temp_dir / 'Brazilian.py'

    build_dir.mkdir(exist_ok=True)

    temp_dir.mkdir(exist_ok=True)

    if not brazilian_file.exists():
        shutil.copy("C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\src\\The\\Brazilian.py", brazilian_file)

    py_filename = temp_dir / isLangExtension(fn).replace('.br', '.py').split("/")[-1] or isLangExtension(fn).replace('.br', '.py').split("\\")[-1]

    try:
        py_filename.write_text(f'''
import sys
sys.path.append("C:/Users/tempe/Desktop/AllSv/BrazilianTestInterpreter/BrazilianProgrammingLanguage/src/The")
from Brazilian import *

text = """{code}"""
if text.strip() == "":
    exit(1)\n
result, error = run("<stdin>", text)
if error:
  print(error.as_string(), file=sys.stderr)''', encoding='utf-8')
    except Exception as e:
        print(f"Error: Invalid file for build: '{py_filename.name}'. Reason: {e}'", file=sys.stderr)
        return

    try:
        pyinstaller_cmd = (
            f'C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\Brazilian\\Scripts\\pyinstaller.exe '
            f'--icon NONE --distpath {build_dir} --onefile {py_filename}'
        )
        process = subprocess.run(pyinstaller_cmd, text=True, stderr=subprocess.PIPE)

        if process.returncode == 0:
            shutil.rmtree(temp_dir)

            spec_file = cwd / isLangExtension(fn).replace('.br', '.spec').split("/")[-1] or isLangExtension(fn).replace('.br', '.spec').split("\\")[-1]
            if spec_file.exists(): os.remove(spec_file.resolve())
            else: pass
            print("Compiled with success.")
        else:
            print(f"Error: {process.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()