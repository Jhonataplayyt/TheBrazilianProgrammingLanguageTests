#pragma once

#ifdef basBR_exports
#define basBR_API __declspec(dllexport)
#else
#define basBR_API __declspec(dllimport)
#endif

extern "C" basBR_API const char* input_key(const char* msg);