from Brazilian import *

sys.stdout.reconfigure(encoding='utf-8')

def main():
	try:
		while True:
			text = input('Brazilian > ')
			if text.strip() == "": continue
			result, error = run('<stdin>', text)

			if error:
				print(error.as_string(), file=sys.stderr)
			elif result:
				real_result = result.elements[0]
				if len(result.elements) != 1:
					real_result = result
				print(repr(real_result))
				global_symbol_table.set("_", real_result)
	except NameError:
		return

if __name__ == "__main__":
	main()