from Brazilian import *

def main():
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

if __name__ == "__main__":
	main()