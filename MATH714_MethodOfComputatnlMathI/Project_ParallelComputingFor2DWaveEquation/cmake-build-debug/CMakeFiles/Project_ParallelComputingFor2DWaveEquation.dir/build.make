# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/flags.make

CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o: CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/flags.make
CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o -c /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/main.cpp

CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/main.cpp > CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.i

CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/main.cpp -o CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.s

# Object files for target Project_ParallelComputingFor2DWaveEquation
Project_ParallelComputingFor2DWaveEquation_OBJECTS = \
"CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o"

# External object files for target Project_ParallelComputingFor2DWaveEquation
Project_ParallelComputingFor2DWaveEquation_EXTERNAL_OBJECTS =

Project_ParallelComputingFor2DWaveEquation: CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/main.cpp.o
Project_ParallelComputingFor2DWaveEquation: CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/build.make
Project_ParallelComputingFor2DWaveEquation: CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Project_ParallelComputingFor2DWaveEquation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/build: Project_ParallelComputingFor2DWaveEquation

.PHONY : CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/build

CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/clean

CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/depend:
	cd /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug /Users/wenxin/CLionProjects/Study/MATH714_MethodOfComputatnlMathI/Project_ParallelComputingFor2DWaveEquation/cmake-build-debug/CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Project_ParallelComputingFor2DWaveEquation.dir/depend
