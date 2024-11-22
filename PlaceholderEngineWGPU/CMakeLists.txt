# CMakeList.txt : CMake project for PlaceholderEngineWGPU, include source and define
# project specific logic here.
#


# Add source to this project's executable.
add_executable (PlaceholderEngineWGPU "PlaceholderEngineWGPU.cpp" "PlaceholderEngineWGPU.h" "webgpu-utils.h" "webgpu-utils.cpp" 
	Application.h
	Application.cpp
	ResourceManager.h
	ResourceManager.cpp
	implementations.cpp)
target_link_libraries(PlaceholderEngineWGPU PRIVATE glfw webgpu glfw3webgpu imgui glm)
if (EMSCRIPTEN)
target_link_options(PlaceholderEngineWGPU PRIVATE
    -sALLOW_MEMORY_GROWTH
    --preload-file "E:/git_projects/PlaceholderEngineWGPU/resources"
)
endif()
target_copy_webgpu_binaries(PlaceholderEngineWGPU)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PlaceholderEngineWGPU PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
