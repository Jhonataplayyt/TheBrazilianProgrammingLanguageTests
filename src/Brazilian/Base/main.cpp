#define basBR_exports

#include <iostream>
#include <cstring>

#ifdef _WIN32
    #include <conio.h>
#else
    #include "./linux-conio/linux_conio.h"
#endif

extern "C" {
    __declspec(dllexport) const char* input_char(const char* msg) {
        printf(msg);

        char ch;

        ch = _getch();

        char* cstr = new char[2];
        cstr[0] = ch;
        cstr[1] = '\0';
        return cstr;
    }
}