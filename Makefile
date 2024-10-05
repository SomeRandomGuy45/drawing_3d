# Define the target
TARGET = libmain_lib
SRCS = main.cpp
HEADERS = main_lib.h

# Define the platform
UNAME_S := $(shell uname -s)

# Compiler settings
CXX = g++
CXXFLAGS = -shared -fPIC -Wall -Wextra -I. --std=c++20

# Platform-specific settings
ifeq ($(UNAME_S), Darwin)  # macOS
    TARGET_LIB = $(TARGET).dylib
    CXXFLAGS += -fvisibility=default -dynamiclib
    LDFLAGS = -install_name @rpath/$(TARGET_LIB)
else ifeq ($(UNAME_S), Linux)  # Linux
    TARGET_LIB = lib$(TARGET).so
else  # Assuming Windows (requires MinGW or similar)
    TARGET_LIB = $(TARGET).dll
    CXX = x86_64-w64-mingw32-g++
    CXXFLAGS = -shared -Wall -Wextra -I.
endif

# Build target
all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET_LIB) $(LDFLAGS)

# Clean up
clean:
	rm -f *.o *.so *.dylib *.dll
    