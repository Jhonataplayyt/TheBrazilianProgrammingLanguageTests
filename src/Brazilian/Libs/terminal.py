import threading
import queue
import re
import sys
import msvcrt
import inspect
import ctypes

stop_event = threading.Event()
rets = queue.Queue()
actual = ""
historic = [""]
pos = -1

def get_fn_name(code):
    function_regex = r"def\s+(\w+)\s*\("
    match = re.search(function_regex, code)
    
    if match:
        return match.group(1)
    else:
        print("Function not found.", file=sys.stderr)

def exec_py_func(code, arg):
    ncode = f"{code}\n\n{get_fn_name(code)}('{arg}')"
    
    exec(ncode)

def mov_terminal():
    global stop_event
    while not stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch()

            if ch == b'\x48':  # Up arrow
                rets.put("1")
            elif ch == b'\x50':  # Down arrow
                rets.put("2")
            elif ch == b'\x1b':  # Escape
                rets.put("3")
                stop_event.set()
                break

def input_exec(name):
    global stop_event
    while not stop_event.is_set():
        print(name, end='')

        try:
            user_input = input('')
            if user_input == '':
                rets.put("3")
                stop_event.set()
                break
            rets.put(user_input)
        except (EOFError, KeyboardInterrupt):
            sys.exit()

def main_loop(code, name):
    global actual, pos, historic, stop_event

    while not stop_event.is_set():
        if not rets.empty():
            ret = rets.get()
            if ret == "1":
                if pos + 1 < len(historic):
                    pos += 1
                    actual = historic[pos]
            elif ret == "2":
                if pos - 1 >= 0:
                    pos -= 1
                    actual = historic[pos]
            elif ret == "3":
                stop_event.set()
                break
            else:
                exec_py_func(code, ret)
                
                actual = ""
                historic.append(ret)
                pos = len(historic) - 1

def run_terminal(func, name):
    try:
        sfunc = inspect.getsource(func)
        sname = name

        key_th = threading.Thread(target=mov_terminal)
        input_th = threading.Thread(target=input_exec, args=(sname,))
        main_th = threading.Thread(target=main_loop, args=(sfunc, sname))

        key_th.start()
        input_th.start()
        main_th.start()

        key_th.join()
        input_th.join()
        main_th.join()

    except (EOFError, KeyboardInterrupt):
        sys.exit()

def test(param):
    print(param)

while True:
    run_terminal(test, 'Test > ')
