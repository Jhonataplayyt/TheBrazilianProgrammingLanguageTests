from Brazilian import *
import dis
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    try:
        fn = sys.argv[1]

        fn = fn.replace("\\", "/")

        with open(isLangExtension(fn), "r") as f:
            code = f.read()

        bytess = compile(
            f'from Brazilian import run\nrun("""{fn}""", """{code}""")',
            '<string>',
            'exec'
        )
        disassembler = '\n'.join([line for line in dis.Bytecode(bytess).dis().splitlines()])

        output_file = (isLangExtension(fn)[:-2] + "bro")
        with open(output_file, "wb") as f_out:
            f_out.write(''.join(f'\\{ord(c):03o}' for c in disassembler).encode('utf-8'))
        
        print(f"Output written to {output_file}")

    except Exception as e:
        try:
            print(f"Error: file or parameter not exists {sys.argv[1]} : '{e}'")
        except:
            print("Error: To work, it needs a file or parameter.")

if __name__ == "__main__":
    main()
