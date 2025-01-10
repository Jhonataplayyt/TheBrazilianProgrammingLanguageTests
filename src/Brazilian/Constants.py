import string, os, math, time, sys, pickle, pydantic, importlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import *

for module in os.listdir(os.path.dirname(__file__)):
    if module == "__init__.py" or module[-3:] != ".py":
        continue
    __import__(module[:-3], locals(), globals())
del module

IMPORT_PATH_NAME = ".path"
if not os.path.isfile(IMPORT_PATH_NAME):
  IMPORT_PATHS = [".", os.getcwd() + "/std"]
  with open(IMPORT_PATH_NAME, "w") as f:
    f.write("\n".join(IMPORT_PATHS))
else:
  with open(IMPORT_PATH_NAME, "r") as f:
    IMPORT_PATHS = list(f.readlines())
DIGITS = '0123456789'
LETTERS = string.ascii_letters
VALID_IDENTIFIERS = LETTERS + DIGITS + "$_"

global_variables = {}