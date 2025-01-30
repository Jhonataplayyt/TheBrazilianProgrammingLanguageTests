from Brazilian import *
import threading
import keyboard
import readline
import queue
import sys
import os

stop_event = threading.Event()

def execbr(text):
    if text.strip() == "":
        return
    result, error = run('<stdin>', text)
    if error:
        print(error.as_string(), file=sys.stderr)
    elif result:
        real_result = result.elements[0] if len(result.elements) == 1 else result
        print(repr(real_result))
        global_symbol_table.set("_", real_result)

actual = ""
rets = queue.Queue()
historic = [""]
pos = -1

def movTerminal(rets):
    while not stop_event.is_set():
        if keyboard.is_pressed('page up'):
            rets.put(1)
        elif keyboard.is_pressed('page down'):
            rets.put(2)
        elif keyboard.is_pressed('esc'):
            rets.put(3)
            stop_event.set()
            break

def input_exec(rets):
    global actual
    while not stop_event.is_set():
        try:
            readline.set_startup_hook(lambda: readline.insert_text(actual))
            try:
                user_input = input("\rBrazilian: ")
                rets.put(user_input)
            finally:
                readline.set_startup_hook()
        except (KeyboardInterrupt, EOFError):
            rets.put(3)
            stop_event.set()
            break

def main(rets):
    global actual, historic, pos
    while not stop_event.is_set():
        if not rets.empty():
            ret = rets.get()
            if ret == 1:
                if pos + 1 < len(historic):
                    pos += 1
                    actual = historic[pos]
            elif ret == 2:
                if pos - 1 >= 0:
                    pos -= 1
                    actual = historic[pos]
            elif ret == 3:
                stop_event.set()
                break
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                execbr(ret)
                actual = ""
                historic.append(ret)
                pos = len(historic) - 1

def runTerminal():
    try:
        key_th = threading.Thread(target=movTerminal, args=(rets,), daemon=True)
        input_th = threading.Thread(target=input_exec, args=(rets,), daemon=True)
        main_th = threading.Thread(target=main, args=(rets,), daemon=True)

        key_th.start()
        input_th.start()
        main_th.start()

        key_th.join()
        input_th.join()
        main_th.join()
    except (KeyboardInterrupt, EOFError):
        stop_event.set()
        sys.exit()

if __name__ == "__main__":
    runTerminal()