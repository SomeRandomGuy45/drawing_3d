# Define the source and output
TARGET = libdrawing_3d
SRC = obj_loader.cpp

# Platform detection
UNAME_S := $(shell uname -s)

# Compiler and standard
CXX = g++
CXXFLAGS = --std=c++20 -fPIC -Wall -Wextra -I.

# Libraries and frameworks
LIBS_LINUX = -lGL -lGLEW -lglfw -lassimp
LIBS_MACOS = -framework OpenGL -lGLEW -lglfw -lassimp
LIBS_WINDOWS = -lopengl32 -lglew32 -lglfw3 -lassimp

# Platform-specific settings
ifeq ($(UNAME_S), Darwin)
    TARGET_LIB = $(TARGET).dylib
    LIBS = $(LIBS_MACOS)
    CXXFLAGS += -fvisibility=default
    LDFLAGS = -dynamiclib -install_name @rpath/$(TARGET_LIB)
else ifeq ($(UNAME_S), Linux)
    LIBS = $(LIBS_LINUX)
    TARGET_LIB = lib$(TARGET).so
    LDFLAGS = -shared
else
    TARGET_LIB = $(TARGET).dll
    LIBS = $(LIBS_WINDOWS)
    CXXFLAGS += -DGLFW_INCLUDE_NONE
    LDFLAGS = -shared
endif

# Build target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET_LIB) $(SRC) $(LDFLAGS) $(LIBS)

# Clean up the build
clean:
	rm -f $(TARGET) $(TARGET_LIB)