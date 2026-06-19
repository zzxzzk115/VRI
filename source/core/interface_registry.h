// InterfaceRegistry - backs vriGetInterface().
//
// Each backend device registers its pre-filled function tables by name. A
// query copies the stored table into the caller's struct after validating the
// requested size against the registered size (guards ABI drift). This is the
// single mechanism behind both the core interface and every optional/extension
// interface (section B/C of the design).
#pragma once

#include <cstddef>
#include <cstring>
#include <string_view>
#include <vector>

#include <vri/vri_base.h>

namespace vri::core
{
    class InterfaceRegistry
    {
    public:
        // The registered table must outlive the registry (backends use static
        // tables), so we store a borrowed pointer, never an owning copy.
        void Register(std::string_view name, const void* table, size_t size)
        {
            m_entries.push_back({name, table, size});
        }

        [[nodiscard]] VriResult Get(std::string_view name, size_t size, void* out) const
        {
            if (out == nullptr)
                return VriResult_InvalidArgument;

            for (const Entry& e : m_entries)
            {
                if (e.name != name)
                    continue;
                if (e.size != size)
                    return VriResult_InvalidArgument; // ABI mismatch
                std::memcpy(out, e.table, e.size);
                return VriResult_Success;
            }
            return VriResult_Unsupported;
        }

    private:
        struct Entry
        {
            std::string_view name;
            const void*      table;
            size_t           size;
        };

        std::vector<Entry> m_entries;
    };
} // namespace vri::core
