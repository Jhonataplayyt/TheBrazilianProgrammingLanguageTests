#pragma once

#ifdef _WIN32
    #ifdef basBR_exports
    #define basBR_API __declspec(dllexport)
    #else
    #define basBR_API __declspec(dllimport)
    #endif
#else
    #ifdef basBR_exports
    #define basBR_API __attribute__((visibility("default")))
    #else
    #define basBR_API __attribute__((visibility("default")))
    #endif
#endif

extern "C" basBR_API const char* input_key(const char* msg);