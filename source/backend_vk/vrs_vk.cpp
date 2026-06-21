// vrs_vk.cpp - Variable Rate Shading via VK_KHR_fragment_shading_rate.
//
// Only the per-draw (pipeline) rate, set dynamically with vkCmdSetFragmentShadingRateKHR.
// pipelineFragmentShadingRate makes that command always-dynamic once enabled, so no
// pipeline state change is needed; the rate applies to subsequent draws.
#include "vrs_vk.h"

#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        CommandBufferVK* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }

        VkExtent2D ToFragmentSize(VriShadingRate rate)
        {
            switch (rate)
            {
                case VriShadingRate_1x1: return {1, 1};
                case VriShadingRate_1x2: return {1, 2};
                case VriShadingRate_2x1: return {2, 1};
                case VriShadingRate_2x2: return {2, 2};
                case VriShadingRate_2x4: return {2, 4};
                case VriShadingRate_4x2: return {4, 2};
                case VriShadingRate_4x4: return {4, 4};
                default:                 return {1, 1};
            }
        }

        VkFragmentShadingRateCombinerOpKHR ToCombiner(VriShadingRateCombiner c)
        {
            switch (c)
            {
                case VriShadingRateCombiner_Keep:    return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
                case VriShadingRateCombiner_Replace: return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;
                case VriShadingRateCombiner_Min:     return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR;
                case VriShadingRateCombiner_Max:     return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR;
                case VriShadingRateCombiner_Sum:     return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_KHR; // "Sum" == accumulate; VK exposes MUL
                default:                             return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
            }
        }

        void VRI_CALL CmdSetShadingRate(VriCommandBuffer* cmd, const VriShadingRateDesc* desc)
        {
            CommandBufferVK* c = CB(cmd);
            const PFN_vkCmdSetFragmentShadingRateKHR fn = c->device->Ext().CmdSetFragmentShadingRate;
            if (!fn || !desc)
                return;
            const VkExtent2D size = ToFragmentSize(desc->shadingRate);
            const VkFragmentShadingRateCombinerOpKHR combiners[2] = {
                ToCombiner(desc->primitiveCombiner),
                ToCombiner(desc->attachmentCombiner),
            };
            fn(c->cmd, &size, combiners);
        }

        const VriShadingRateInterface g_vrsVK = {
            CmdSetShadingRate,
        };
    } // namespace

    const VriShadingRateInterface* GetShadingRateInterfaceVK() { return &g_vrsVK; }
} // namespace vri::vk
