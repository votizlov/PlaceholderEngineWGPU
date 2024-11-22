// PlaceholderEngineWGPU.cpp : Defines the entry point for the application.
//

#include "PlaceholderEngineWGPU.h"

#include "Application.h"

int main(int, char**) {
	Application app;
	if (!app.onInit()) return 1;

#ifdef __EMSCRIPTEN__
	// Equivalent of the main loop when using Emscripten:
	auto callback = [](void* arg) {
		Application* pApp = reinterpret_cast<Application*>(arg);
		pApp->onFrame(); // 4. We can use the application object
		};
	emscripten_set_main_loop_arg(callback, &app, 0, true);
#else // __EMSCRIPTEN__

	while (app.isRunning()) {
		app.onFrame();
	}
#endif // __EMSCRIPTEN__

	app.onFinish();
	return 0;
}
/*
#include <webgpu/webgpu.h>
#include "webgpu-utils.h"

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif // __EMSCRIPTEN__

using namespace std;
#include <iostream>

int main()
{
	WGPUInstanceDescriptor desc = {};
	desc.nextInChain = nullptr;

#ifdef WEBGPU_BACKEND_EMSCRIPTEN
	WGPUInstance instance = wgpuCreateInstance(nullptr);
#else //  WEBGPU_BACKEND_EMSCRIPTEN
	WGPUInstance instance = wgpuCreateInstance(&desc);
#endif //  WEBGPU_BACKEND_EMSCRIPTEN

	if (!instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return 1;
	}

	std::cout << "WGPU instance: " << instance << std::endl;

	std::cout << "Requesting adapter..." << std::endl;
	WGPURequestAdapterOptions adapterOpts = {};
	adapterOpts.nextInChain = nullptr;
	WGPUAdapter adapter = requestAdapterSync(instance, &adapterOpts);
	std::cout << "Got adapter: " << adapter << std::endl;

	// Display some information about the adapter
	inspectAdapter(adapter);

	// We no longer need to use the instance once we have the adapter
	wgpuInstanceRelease(instance);

	wgpuAdapterRelease(adapter);

	return 0;
}*/