#pragma once
#include "BaseComponent.h"
#include <vector>
#include <webgpu/webgpu.hpp>
#include <glm/glm.hpp>

struct MeshData {
    wgpu::Buffer vertexBuffer;
    size_t vertexCount;

    // Default constructor
    MeshData()
        : vertexBuffer(nullptr)
        , vertexCount(0)
    {
    }
};

class MeshRendererComponent : public BaseComponent {
public:
    std::vector<MeshData> meshes;
    // Optional: store transform or material info here
    glm::mat4 localTransform = glm::mat4(1.0f);
};