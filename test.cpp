#include <iostream>
#include "Python.h"

int main() {
    Py_Initialize();

    const char* pythonCode = "def printTerm(param):\n    print(param)\n\nprintTerm('Oi')";

    PyRun_SimpleString(pythonCode);

    Py_Finalize();

    return 0;
}