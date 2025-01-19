#pragma once
#include "BaseSystem.h"
#include "MeshRendererComponent.h"
#include "ResourceManager.h"
#include <webgpu/webgpu.hpp>

class MeshRenderSystem : public BaseSystem {
public:
    // Array of components can be managed externally or within the system
    std::vector<MeshRendererComponent*> meshRenderers;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroup bindGroup;

    MeshRenderSystem(wgpu::RenderPipeline p, wgpu::BindGroup bg)
        : pipeline(p), bindGroup(bg) {
    }

    MeshRenderSystem()
        : pipeline(nullptr), bindGroup(nullptr) {
    }

    void Update() override {
        // Example: pseudo code for drawing all mesh renderer components
        // (You would call this from Application::onFrame's render pass)
    }

    void RenderAll(wgpu::RenderPassEncoder& pass) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup, 0, nullptr);
        //std::cout << "drawing meshes: " << meshRenderers.size() << std::endl;
        for (auto* mr : meshRenderers) {
            for (auto& meshData : mr->meshes) {
                pass.setVertexBuffer(0, meshData.vertexBuffer, 0, meshData.vertexCount * sizeof(ResourceManager::VertexAttributes));
                pass.draw(meshData.vertexCount, 1, 0, 0);
            }
        }
    }
};