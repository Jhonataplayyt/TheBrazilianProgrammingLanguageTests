#define basBR_exports

#include <iostream>
#include <thread>
#include <queue>
#include <string>
#include <atomic>
#include <vector>
#include <conio.h>
#include <cstdlib>
#include <csignal>
#include <regex>
#include <Python.h>

std::atomic<bool> stop_event(false);
std::queue<std::string> rets;
std::string actual = "";
std::vector<std::string> historic = {""};
int pos = -1;

std::string getFnName(std::string code) {
    std::regex function_regex(R"(def\s+(\w+)\s*\()");
    std::smatch match;

    if (std::regex_search(code, match, function_regex)) {
        return match[1];
    } else {
        std::cerr << "Function not finded." << std::endl;
    }
}

void execPyFunc(const std::string& code, const std::string& arg) {
    Py_Initialize();

    std::string ncode = code + "\n\n" + (getFnName(code) + "(\'" + arg + "\')");

    std::cout << ncode << std::endl;

    PyRun_SimpleString(ncode.c_str());
    
    Py_Finalize();
}

void movTerminal() {
    while (!stop_event.load()) {
        if (_kbhit()) {
            char ch = _getch();
            if (ch == 72) {
                rets.push("1");
            } else if (ch == 80) {
                rets.push("2");
            } else if (ch == 27) {
                rets.push("3");
                stop_event.store(true);
                break;
            }
        }
    }
}

void input_exec(std::string name) {
    while (!stop_event.load()) {
        std::string user_input;
        std::cout << name;
        std::getline(std::cin, user_input);
        if (std::cin.eof()) {
            rets.push("3");
            stop_event.store(true);
            break;
        }
        rets.push(user_input);
    }
}

void main_loop(const std::string& code, const std::string& name) {
    while (!stop_event.load()) {
        if (!rets.empty()) {
            std::string ret = rets.front();
            rets.pop();
            if (ret == "1") {
                if (pos + 1 < historic.size()) {
                    pos++;
                    actual = historic[pos];
                }
            } else if (ret == "2") {
                if (pos - 1 >= 0) {
                    pos--;
                    actual = historic[pos];
                }
            } else if (ret == "3") {
                stop_event.store(true);
                break;
            } else {
                execPyFunc(code, ret);

                std::cout << name;
                actual = "";
                historic.push_back(ret);
                pos = historic.size() - 1;
            }
        }
    }
}

extern "C" {
    __declspec(dllexport) void runTerminal(const char* func, const char* name) {
        std::string sfunc(func);
        std::string sname(name);

        std::thread key_th(movTerminal);
        std::thread input_th(input_exec, sname);
        std::thread main_th(main_loop, sfunc, sname);

        key_th.join();
        input_th.join();
        main_th.join();
    }
}