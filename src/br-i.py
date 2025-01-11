import string, os, math, time, sys, pickle
from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from Brazilian import *

def main():
  try:
    fn = sys.argv[1]
    with open(isLangExtension(fn), "r") as f:
      code = open(fn, 'r').read()
    _, error = run(fn, code)
    if error:
      print(error.as_string(), file=sys.stderr)
      pass
    pass
  except Exception as e:
    print(e)

    try:
      print(f"Error: file or parameter not exists {sys.argv[1]}")
    except:
      print(f"Error: To work, it needs a file or parameter.")

if __name__ == "__main__":
	main()