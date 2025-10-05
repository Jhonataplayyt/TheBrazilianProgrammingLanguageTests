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

#ifdef __cplusplus
extern "C" {
#endif

basBR_API const char* input_char(const char* msg);

#ifdef __cplusplus
}
#endif
