#include <iostream>
#include <conio.h>
#include <string>
#include <Python.h>

#define PY_SSIZE_T_CLEAN

using namespace std;

string input_key(string val = "") {
    cout << val;
    char ret = _getch();
    cout << ret << endl;
    return string(1, ret);
}

static PyObject *Input_key(PyObject *self, PyObject *args) {
    const char *val = nullptr;
    string sts;

    if (!PyArg_ParseTuple(args, "s", &val)) {
        return NULL;
    }

    sts = input_key(val);
    return PyUnicode_FromString(sts.c_str());
}

static PyMethodDef BasBR_mod[] = {
    {"input_key", Input_key, METH_VARARGS, "Returns a one character."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef basBR = {
    PyModuleDef_HEAD_INIT,
    "basBR",
    NULL,
    -1,
    BasBR_mod
};

PyMODINIT_FUNC PyInit_basBR(void) {
    return PyModule_Create(&basBR);
}