from Brazilian import *
import shutil
import subprocess
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def main():
    try:
        fn = None

        fn = sys.argv[1] if len(sys.argv) > 1 else ""

        code = open(isLangExtension(fn), 'r', encoding='utf-8').read()

        cwd = Path.cwd()
        build_dir = cwd / 'build'
        temp_dir = cwd / 'temp'
        brazilian = temp_dir / "Brazilian"

        build_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)

        if not brazilian.exists():
            shutil.copytree("C:\\Users\\tempe\\Desktop\\AllSv\\BrazilianTestInterpreter\\BrazilianProgrammingLanguage\\src\\Brazilian", brazilian)

        py_filename = temp_dir / (isLangExtension(fn)[:-2] + "py").split("/")[-1] or (isLangExtension(fn)[:-2] + "py").split("\\")[-1]

        try:
            with open(str(os.path.abspath(py_filename)), "w", encoding='utf-8') as f:
                f.write(f'''
import sys
import os
sys.path.append(os.path.abspath("C:\\\\Users\\\\tempe\\\\Desktop\\\\AllSv\\\\BrazilianTestInterpreter\\\\BrazilianProgrammingLanguage\\\\src\\\\Brazilian"))

from Brazilian import run

text = """{code}"""
if text.strip() == "":
    sys.exit(1)

try:
    result, error = run("<stdin>", text)
    if error:
        print(error.as_string(), file=sys.stderr)
except NameError:
    sys.exit()
''')
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

                spec_file = cwd / (isLangExtension(fn)[:-2] + "spec").split("/")[-1] or (isLangExtension(fn)[:-2] + "spec").split("\\")[-1]
                if spec_file.exists(): os.remove(spec_file.resolve())
                else: pass
                print("Compiled with success.")
            else:
                print(f"Error: {process.stderr}", file=sys.stderr)
        except Exception as e:
                print(f"Compilation error: {e}", file=sys.stderr)
    except:
        print("Error: We need a file for the compiler to work.", file=sys.stderr)
        return


if __name__ == "__main__":
    main()