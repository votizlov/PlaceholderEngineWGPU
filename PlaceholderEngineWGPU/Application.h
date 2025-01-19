#pragma once
#include <webgpu/webgpu.hpp>
#include <glm/glm.hpp>

#include <array>
//#include "BaseSystem.h"
#include "MeshRendererSystem.h"
#include "cgltf.h"

// Forward declare
struct GLFWwindow;

class Application {
public:
	// A function called only once at the beginning. Returns false is init failed.
	bool onInit();

	// A function called at each frame, guaranteed never to be called before `onInit`.
	void onFrame();

	// A function called only once at the very end.
	void onFinish();

	// A function that tells if the application is still running.
	bool isRunning();

	// A function called when the window is resized.
	void onResize();

	// Mouse events
	void onMouseMove(double xpos, double ypos);
	void onMouseButton(int button, int action, int mods);
	void onScroll(double xoffset, double yoffset);

private:
	bool initWindowAndDevice();
	void terminateWindowAndDevice();

	bool initSwapChain();
	void terminateSwapChain();

	bool initDepthBuffer();
	void terminateDepthBuffer();

	bool initRenderPipeline();
	void terminateRenderPipeline();

	bool initTexture();
	void terminateTexture();

	bool initGeometry();
	void FillVertexData(cgltf_primitive& prim, std::vector<ResourceManager::VertexAttributes>& vertices);
	void loadFourareenTest();
	bool loadGltfScene(const char* path, wgpu::Device device, wgpu::Queue queue, MeshRenderSystem& meshRenderSystem);
	void terminateGeometry();

	bool initUniforms();
	void terminateUniforms();

	bool initLightingUniforms();
	void terminateLightingUniforms();
	void updateLightingUniforms();

	bool initBindGroupLayout();
	void terminateBindGroupLayout();

	bool initBindGroup();
	void terminateBindGroup();

	void updateProjectionMatrix();
	void updateViewMatrix();
	void updateDragInertia();

	bool initGui(); // called in onInit
	void terminateGui(); // called in onFinish
	void updateGui(wgpu::RenderPassEncoder renderPass); // called in onFrame
	//wgpu::TextureView GetNextSurfaceTextureView();

private:
	// (Just aliases to make notations lighter)
	using mat4x4 = glm::mat4x4;
	using vec4 = glm::vec4;
	using vec3 = glm::vec3;
	using vec2 = glm::vec2;

	/**
	 * The same structure as in the shader, replicated in C++
	 */
	struct MyUniforms {
		// We add transform matrices
		mat4x4 projectionMatrix;
		mat4x4 viewMatrix;
		mat4x4 modelMatrix;
		vec4 color;
		float time;
		float _pad[3];
	};
	// Have the compiler check byte alignment
	static_assert(sizeof(MyUniforms) % 16 == 0);

	struct LightingUniforms {
		std::array<vec4, 2> directions;
		std::array<vec4, 2> colors;
	};
	static_assert(sizeof(LightingUniforms) % 16 == 0);

	struct CameraState {
		// angles.x is the rotation of the camera around the global vertical axis, affected by mouse.x
		// angles.y is the rotation of the camera around its local horizontal axis, affected by mouse.y
		vec2 angles = { 0.8f, 0.5f };
		// zoom is the position of the camera along its local forward axis, affected by the scroll wheel
		float zoom = -1.2f;
	};

	struct DragState {
		// Whether a drag action is ongoing (i.e., we are between mouse press and mouse release)
		bool active = false;
		// The position of the mouse at the beginning of the drag action
		vec2 startMouse;
		// The camera state at the beginning of the drag action
		CameraState startCameraState;

		// Constant settings
		float sensitivity = 0.01f;
		float scrollSensitivity = 0.1f;

		// Inertia
		vec2 velocity = { 0.0, 0.0 };
		vec2 previousDelta;
		float intertia = 0.9f;
	};

	// Window and Device
	GLFWwindow* m_window = nullptr;
	wgpu::Instance m_instance = nullptr;
	wgpu::Surface m_surface = nullptr;
	wgpu::Device m_device = nullptr;
	wgpu::Queue m_queue = nullptr;
	wgpu::TextureFormat m_swapChainFormat = wgpu::TextureFormat::Undefined;
	// Keep the error callback alive
	std::unique_ptr<wgpu::ErrorCallback> m_errorCallbackHandle;

	// Swap Chain
	wgpu::SwapChain m_swapChain = nullptr;
	//wgpu::Surface m_surface = nullptr;

	// Depth Buffer
	wgpu::TextureFormat m_depthTextureFormat = wgpu::TextureFormat::Depth24Plus;
	wgpu::Texture m_depthTexture = nullptr;
	wgpu::TextureView m_depthTextureView = nullptr;

	// Render Pipeline
	wgpu::ShaderModule m_shaderModule = nullptr;
	wgpu::RenderPipeline m_pipeline = nullptr;

	// Texture
	wgpu::Sampler m_sampler = nullptr;
	wgpu::Texture m_texture = nullptr;
	wgpu::TextureView m_textureView = nullptr;

	// Geometry
	wgpu::Buffer m_vertexBuffer = nullptr;
	int m_vertexCount = 0;

	// Uniforms
	wgpu::Buffer m_uniformBuffer = nullptr;
	MyUniforms m_uniforms;
	wgpu::Buffer m_lightingUniformBuffer = nullptr;
	LightingUniforms m_lightingUniforms;
	bool m_lightingUniformsChanged = true;

	// Bind Group Layout
	wgpu::BindGroupLayout m_bindGroupLayout = nullptr;

	// Bind Group
	wgpu::BindGroup m_bindGroup = nullptr;

	CameraState m_cameraState;
	DragState m_drag;

	std::vector<BaseSystem*> registeredSystems;
	MeshRenderSystem* m_meshRenderSystem;

};