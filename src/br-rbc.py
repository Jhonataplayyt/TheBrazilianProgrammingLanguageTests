import sys
import re
from Brazilian import *

sys.stdout.reconfigure(encoding='utf-8')

def isLangExtension(x):
    if x.endswith(".bro"):
        return x
    else:
        print("Extension Error: The extension needs to be '.bro', not '." + x.split('.')[1] + "'")
        exit(1)

def re_assembly(dis_output):
    lines = dis_output.strip().split("\n")
    python_code = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("0 "):
            continue
        
        if "IMPORT_NAME" in line and "IMPORT_FROM" in line:
            match = re.search(r"IMPORT_NAME\s+\d+\s+\((.+)\)", line)
            if match:
                module_name = match.group(1)
                next_line = next(iter(lines), "").strip()
                match_next = re.search(r"IMPORT_FROM\s+\d+\s+\((.+)\)", next_line)
                if match_next:
                    func_name = match_next.group(1)
                    python_code.append(f"from {module_name} import {func_name}")
            continue
        
        elif "LOAD_CONST" in line:
            match = re.search(r"LOAD_CONST\s+\d+\s+\((.+)\)", line)
            if match:
                const_value = match.group(1).strip("'")
                if "\\n" in const_value:
                    const_value = const_value.replace("\\n", "\n")
                python_code.append(const_value)
            continue
        
        elif "LOAD_NAME" in line:
            match = re.search(r"LOAD_NAME\s+\d+\s+\((.+)\)", line)
            if match:
                python_code.append(match.group(1))
            continue

        elif "CALL" in line:
            match = re.search(r"CALL\s+\d+", line)
            if match:
                python_code.append("()")
            continue
        
        elif "RETURN_CONST" in line:
            match = re.search(r"RETURN_CONST\s+\d+\s+\((.+)\)", line)
            if match:
                python_code.append(f"return {match.group(1)}")
            continue

    return "\n".join(python_code)

def engRevBin(x):
    clsBytes = x.replace('\\', '')
    try:
        bytes_array = bytearray(int(clsBytes[i:i+3], 8) for i in range(0, len(clsBytes), 3))
        return bytes_array.decode('utf-8')
    except Exception as e:
        print(f"Error decoding binary content: {e}")
        return ""

def main():
    try:
        fn = sys.argv[1]
        fn = fn.replace("\\", "/")

        fc = open(isLangExtension(fn), 'rb').read().decode('utf-8', errors='ignore')

        out = re_assembly(engRevBin(fc))

        pattern = r"{}\.br\n(.*?)\n\(\)".format(isLangExtension(fn)[:-4])

        matches = re.findall(pattern, out, re.DOTALL)

        try:
            code_block = matches[0]

            result, error = run('<stdin>', code_block)
            
            if error:
                print(error.as_string(), file=sys.stderr)
        except Exception as e:
            print(f"Intern Error: {e}")
    except:
        try:
            print(f"Error: file or parameter not exists {sys.argv[1]}")
        except:
            try:
                print(f"Error: To work, it needs a file or parameter.")
            except NameError:
                return

if __name__ == '__main__':
    main()
