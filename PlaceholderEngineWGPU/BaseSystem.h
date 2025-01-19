#pragma once
#include <vector>

class BaseSystem {
public:
    virtual ~BaseSystem() = default;
    virtual void Update() = 0;
};