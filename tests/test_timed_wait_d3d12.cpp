#include "backend_d3d12/core_d3d12.h"
#include "backend_d3d12/objects_d3d12.h"

#include <doctest/doctest.h>

namespace
{
    // Count native event registrations without requiring an adapter or GPU work.
    class CountingFence final : public ID3D12Fence
    {
    public:
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override
        {
            if (!object)
            {
                return E_POINTER;
            }
            *object = nullptr;
            if (riid != __uuidof(IUnknown) && riid != __uuidof(ID3D12Fence))
            {
                return E_NOINTERFACE;
            }
            *object = static_cast<ID3D12Fence*>(this);
            AddRef();
            return S_OK;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++m_refCount; }
        ULONG STDMETHODCALLTYPE Release() override
        {
            const ULONG remaining = --m_refCount;
            if (!remaining)
            {
                delete this;
            }
            return remaining;
        }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void** device) override
        {
            if (!device)
            {
                return E_POINTER;
            }
            *device = nullptr;
            return E_NOINTERFACE;
        }
        UINT64 STDMETHODCALLTYPE  GetCompletedValue() override { return m_completed; }
        HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 value, HANDLE event) override
        {
            ++m_registrations;
            if (m_completeOnRegistration)
            {
                m_completed = value;
                return SetEvent(event) ? S_OK : E_FAIL;
            }
            return S_OK;
        }
        HRESULT STDMETHODCALLTYPE Signal(UINT64 value) override
        {
            m_completed = value;
            return S_OK;
        }
        uint32_t Registrations() const { return m_registrations; }
        void     CompleteOnRegistration() { m_completeOnRegistration = true; }

    private:
        ULONG    m_refCount               = 1;
        UINT64   m_completed              = 7;
        uint32_t m_registrations          = 0;
        bool     m_completeOnRegistration = false;
    };
} // namespace

TEST_CASE("D3D12 finite waits do not leave event registrations pending")
{
    auto*                  native = new CountingFence;
    vri::d3d12::FenceD3D12 fence;
    fence.fence.Attach(native);
    auto*       handle = vri::d3d12::ToHandle(&fence);
    const auto* core   = vri::d3d12::GetCoreInterfaceD3D12();
    for (uint32_t attempt = 0; attempt < 8; ++attempt)
    {
        CHECK(core->WaitForFence(handle, 8, 10000) == VriResult_Timeout);
        CHECK(core->GetFenceValue(handle) == 7);
    }
    CHECK(native->Registrations() == 0);
    native->CompleteOnRegistration();
    CHECK(core->WaitForFence(handle, 8, UINT64_MAX) == VriResult_Success);
    CHECK(core->GetFenceValue(handle) == 8);
    CHECK(native->Registrations() == 1);
    REQUIRE(native->Signal(UINT64_MAX) == S_OK);
    CHECK(core->WaitForFence(handle, 9, 10000) == VriResult_DeviceLost);
    CHECK(native->Registrations() == 1);
}
