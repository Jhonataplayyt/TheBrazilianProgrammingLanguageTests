#define basBR_exports

#include <iostream>
#include <cstring>

#ifdef _WIN32
    #include <conio.h>
#else
    #include <termios.h>
    #include <unistd.h>

    char getch() {
        char buf = 0;

        struct termios old = {};

        if (tcgetattr(STDIN_FILENO, &old) < 0) perror("tcgetattr()");
    
        struct termios new_t = old;

        new_t.c_lflag &= ~(ICANON | ECHO);

        if (tcsetattr(STDIN_FILENO, TCSANOW, &new_t) < 0) perror("tcsetattr ICANON");

        if (read(STDIN_FILENO, &buf, 1) < 0) perror("read()");
    
        if (tcsetattr(STDIN_FILENO, TCSADRAIN, &old) < 0) perror("tcsetattr ~ICANON");

        return buf;
    }
#endif

extern "C" {
    #ifdef _WIN32
        __declspec(dllexport) const char* input_char(const char* msg) {
            printf("%s", msg);

            char ch;

            ch = _getch();

            char* cstr = new char[2];
            cstr[0] = ch;
            cstr[1] = '\0';
            return cstr;
        }
    #else
        __attribute__((visibility("default"))) const char* input_char(const char* msg) {
            printf("%s", msg);

            char ch;

            ch = getch();

            char* cstr = new char[2];
            cstr[0] = ch;
            cstr[1] = '\0';
            return cstr;
        }
    #endif
}