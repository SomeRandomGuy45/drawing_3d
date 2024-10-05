#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <vector>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <map>
#include <cmath>
#include <thread>

#if defined(__APPLE__) // macOS
#include <dispatch/dispatch.h>
#define runOnMainThread(block) \
    dispatch_async(dispatch_get_main_queue(), block)
#elif defined(_WIN32) // Windows
#include <windows.h>
#define WM_RUN_ON_MAIN_THREAD (WM_USER + 1)

inline void runOnMainThread(void (*func)()) {
    HWND hwnd = GetConsoleWindow(); // Use your window handle
    if (hwnd) {
        PostMessage(hwnd, WM_RUN_ON_MAIN_THREAD, reinterpret_cast<WPARAM>(func), 0);
    }
}

// The window procedure to handle messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_RUN_ON_MAIN_THREAD) {
        auto func = reinterpret_cast<void(*)()>(wParam);
        if (func) {
            func(); // Call the function on the main thread
        }
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

#else // Linux (or other platforms)
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <mutex>

std::queue<void (*)()> mainThreadQueue;
std::mutex queueMutex;

void processMainThreadQueue() {
    std::lock_guard<std::mutex> lock(queueMutex);
    while (!mainThreadQueue.empty()) {
        auto func = mainThreadQueue.front();
        mainThreadQueue.pop();
        func(); // Execute the function
    }
}

#define runOnMainThread(func) \
    { \
        std::lock_guard<std::mutex> lock(queueMutex); \
        mainThreadQueue.push(func); \
    }

#endif

std::thread windowThread;

#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
    #define helper __stdcall
    #define open extern "C"
#else
    #define DLLEXPORT __attribute__((visibility("default")))
    #define helper
    #define open
#endif

// Function pointer type for library functions
typedef std::string (*FunctionPtr)(const std::vector<std::string>& args);

extern "C" open DLLEXPORT std::string helper createWin(const std::vector<std::string>& args);

// Function to list all available functions in the library
extern "C" open DLLEXPORT std::vector<std::string> helper listFunctions();

// Function to get a function pointer by name
extern "C" DLLEXPORT FunctionPtr helper getFunction(const char* name);

#endif