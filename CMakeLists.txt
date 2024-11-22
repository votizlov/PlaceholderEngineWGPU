# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (
	"PlaceholderEngineWGPU"
	VERSION 0.1.0
)

option(DEV_MODE "Set up development helper settings" ON)

if (NOT EMSCRIPTEN)
	add_subdirectory(glfw)
else()
	# Emscripten has built-in support for GLFW but requires the `-sUSE_GLFW=3` link option:
	add_library(glfw INTERFACE)
	target_link_options(glfw INTERFACE -sUSE_GLFW=3)
endif()

# Include sub-projects.
add_subdirectory(webgpu)
add_subdirectory(glfw3webgpu)
add_subdirectory(imgui)
add_subdirectory(glm)
add_subdirectory ("PlaceholderEngineWGPU")

if(DEV_MODE)
	# In dev mode, we load resources from the source tree, so that when we
	# dynamically edit resources (like shaders), these are correctly
	# versionned.
	target_compile_definitions(PlaceholderEngineWGPU PRIVATE
		RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
	)
else()
	# In release mode, we just load resources relatively to wherever the
	# executable is launched from, so that the binary is portable
	target_compile_definitions(PlaceholderEngineWGPU PRIVATE
		RESOURCE_DIR="./resources"
	)
endif()

if (EMSCRIPTEN)
    # Add Emscripten-specific link options
    target_link_options(PlaceholderEngineWGPU PRIVATE
        -sUSE_GLFW=3 # Use Emscripten-provided GLFW
        -sUSE_WEBGPU # Handle WebGPU symbols
        -sASYNCIFY # Required by WebGPU-C++
    )
    set_target_properties(PlaceholderEngineWGPU PROPERTIES SUFFIX ".html")
endif()
