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
#include "object_registry.h"

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

        // Lives here rather than in each backend so EnumerateObjects means the same thing on all
        // of them and a new backend cannot ship a silently partial listing. The backend supplies
        // only the size, which is the one part that cannot be shared - VMA's allocation info,
        // D3D12's GetResourceAllocationInfo, Metal's allocatedSize.
        ObjectRegistry&       Objects() { return m_objects; }
        const ObjectRegistry& Objects() const { return m_objects; }

    protected:
        InterfaceRegistry m_registry;
        ObjectRegistry    m_objects;
    };
} // namespace vri::core
