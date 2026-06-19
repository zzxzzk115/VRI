// DeviceBase - backend-agnostic base for every VriDevice implementation.
//
// A backend device derives from this, fills the InterfaceRegistry in its
// constructor (always registering the core interface, conditionally
// registering optional interfaces), and is handed back to callers as an opaque
// VriDevice*. vriDestroyDevice() deletes through the virtual destructor.
#pragma once

#include <vri/vri_base.h>
#include <vri/vri_structs.h>

#include "interface_registry.h"

namespace vri::core
{
    class DeviceBase
    {
    public:
        virtual ~DeviceBase() = default;

        [[nodiscard]] VriResult GetInterface(std::string_view name, size_t size, void* out) const
        {
            return m_registry.Get(name, size, out);
        }

    protected:
        InterfaceRegistry m_registry;
    };
} // namespace vri::core
