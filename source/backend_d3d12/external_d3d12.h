// external_d3d12.h - D3D12 external memory/fence export interface (shared committed
// resources via CreateSharedHandle + shared ID3D12Fence).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the external-memory/fence export function table. Registered by the device only
    // when VriFeature_ExternalMemory was granted at creation.
    const VriExternalInterface* GetExternalInterfaceD3D12();
} // namespace vri::d3d12
