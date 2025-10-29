from Brazilian import *

sys.stdout.reconfigure(encoding='utf-8')

def main():
  #try:
    fn = "/storage/emulated/0/TheBrazilianProgrammingLanguageTests/expls/temp.br"
    code = open(fn, 'r', encoding='utf-8').read()
    _, error = run(fn, code)
    if error:
      print(error.as_string(), file=sys.stderr)
      pass
    pass
  #except Exception as e:
  #  try:
  #    try:
  #      print(f"Error: file or parameter not exists {sys.argv[1]}")
  #    except:
  #      print(f"Error: To work, it needs a file or parameter.")
  #  except NameError:
  #    return

if __name__ == "__main__":
	main()