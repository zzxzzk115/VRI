        // core_d3d12_pipeline.inc - pipeline layouts (root signatures), graphics/compute/mesh
        // pipeline creation and descriptor pools/sets for the D3D12 backend. #included into
        // core_d3d12.cpp (one translation unit / anonymous namespace); not compiled on its own.
        VriResult VRI_CALL CreatePipelineLayout(VriDevice*                   device,
                                                const VriPipelineLayoutDesc* desc,
                                                VriPipelineLayout**          out)
        {
            DeviceD3D12*         d = Dev(device);
            PipelineLayoutD3D12* l = new PipelineLayoutD3D12 {};
            l->device              = d;
            // Constant buffers -> root CBV (by GPU address). Textures (SRV) and samplers ->
            // one descriptor table per set per heap-type. D3D12 registers are per-type, so we
            // flatten per-type from 0 within the set (matching Slang's HLSL assignment): the
            // first texture -> t0, first sampler -> s0, etc.
            std::vector<D3D12_ROOT_PARAMETER>                params;
            std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> rangeStore; // keep ranges alive for serialize
            rangeStore.reserve(desc->descriptorSetNum * 2);
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& sd = desc->descriptorSets[s];
                LayoutSetD3D12              setInfo {};
                setInfo.set                = sd.registerSpace;
                uint32_t            srvIdx = 0, uavIdx = 0, samplerIdx = 0;
                std::vector<size_t> srvBindings, uavBindings, samplerBindings; // indices into l->bindings to patch
                auto                isUav = [](VriDescriptorType t) { return t == VriDescriptorType_StorageTexture; };
                for (uint32_t r = 0; r < sd.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& rd = sd.ranges[r];
                    const uint32_t                n  = rd.descriptorNum ? rd.descriptorNum : 1u;
                    if (rd.descriptorType == VriDescriptorType_ConstantBuffer ||
                        rd.descriptorType == VriDescriptorType_StorageBuffer)
                    {
                        // Constant buffer -> root CBV; storage (RW) buffer -> root UAV.
                        D3D12_ROOT_PARAMETER p      = {};
                        p.ParameterType             = rd.descriptorType == VriDescriptorType_ConstantBuffer ?
                                                          D3D12_ROOT_PARAMETER_TYPE_CBV :
                                                          D3D12_ROOT_PARAMETER_TYPE_UAV;
                        p.Descriptor.ShaderRegister = rd.baseRegister;
                        p.Descriptor.RegisterSpace  = sd.registerSpace;
                        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
                        l->bindings.push_back({sd.registerSpace,
                                               rd.baseRegister,
                                               rd.descriptorType,
                                               static_cast<uint32_t>(params.size()),
                                               0});
                        params.push_back(p);
                    }
                    else if (rd.descriptorType == VriDescriptorType_Sampler)
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, samplerIdx});
                        samplerBindings.push_back(l->bindings.size() - 1);
                        samplerIdx += n;
                    }
                    else if (isUav(rd.descriptorType)) // StorageTexture -> UAV (u#) in the set's table
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, uavIdx});
                        uavBindings.push_back(l->bindings.size() - 1);
                        uavIdx += n;
                    }
                    else // Texture / StructuredBuffer / AccelerationStructure -> SRV (t#) in the set's table
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, srvIdx});
                        srvBindings.push_back(l->bindings.size() - 1);
                        srvIdx += n;
                    }
                }
                // SRV + UAV share one descriptor table (one root param). Heap layout (APPEND):
                // SRV slots [0, srvIdx) then UAV slots [srvIdx, srvIdx+uavIdx). UAV registers
                // (u#) restart at 0, so the UAV range's BaseShaderRegister is 0 too.
                if (srvIdx > 0 || uavIdx > 0)
                {
                    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
                    if (srvIdx > 0)
                    {
                        D3D12_DESCRIPTOR_RANGE rg            = {};
                        rg.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                        rg.NumDescriptors                    = srvIdx;
                        rg.BaseShaderRegister                = 0;
                        rg.RegisterSpace                     = sd.registerSpace;
                        rg.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                        ranges.push_back(rg);
                    }
                    if (uavIdx > 0)
                    {
                        D3D12_DESCRIPTOR_RANGE rg            = {};
                        rg.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                        rg.NumDescriptors                    = uavIdx;
                        rg.BaseShaderRegister                = 0;
                        rg.RegisterSpace                     = sd.registerSpace;
                        rg.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                        ranges.push_back(rg);
                    }
                    rangeStore.push_back(std::move(ranges));
                    D3D12_ROOT_PARAMETER p                = {};
                    p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(rangeStore.back().size());
                    p.DescriptorTable.pDescriptorRanges   = rangeStore.back().data();
                    p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
                    setInfo.srvTableParam                 = static_cast<int>(params.size());
                    setInfo.srvCount                      = srvIdx + uavIdx; // total heap slots for the table
                    for (size_t bi : srvBindings)
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                    for (size_t bi : uavBindings)
                    {
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                        l->bindings[bi].heapOffset += srvIdx;
                    }
                    params.push_back(p);
                }
                if (samplerIdx > 0)
                {
                    std::vector<D3D12_DESCRIPTOR_RANGE> ranges(1);
                    ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
                    ranges[0].NumDescriptors                    = samplerIdx;
                    ranges[0].BaseShaderRegister                = 0;
                    ranges[0].RegisterSpace                     = sd.registerSpace;
                    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                    rangeStore.push_back(std::move(ranges));
                    D3D12_ROOT_PARAMETER p                = {};
                    p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    p.DescriptorTable.NumDescriptorRanges = 1;
                    p.DescriptorTable.pDescriptorRanges   = rangeStore.back().data();
                    p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
                    setInfo.samplerTableParam             = static_cast<int>(params.size());
                    setInfo.samplerCount                  = samplerIdx;
                    for (size_t bi : samplerBindings)
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                    params.push_back(p);
                }
                l->sets.push_back(setInfo);
            }
            // Push constants -> a root 32-bit-constants parameter at the requested register.
            // Slang lowers [[vk::push_constant]] to an HLSL cbuffer at that b# register.
            if (desc->pushConstantNum > 0)
            {
                const VriPushConstantDesc& pc = desc->pushConstants[0];
                D3D12_ROOT_PARAMETER       p  = {};
                p.ParameterType               = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
                p.Constants.ShaderRegister    = pc.baseRegister;
                p.Constants.RegisterSpace     = 0;
                p.Constants.Num32BitValues    = (pc.size + 3u) / 4u;
                p.ShaderVisibility            = D3D12_SHADER_VISIBILITY_ALL;
                l->hasPush                    = true;
                l->pushRootParam              = static_cast<uint32_t>(params.size());
                l->push32Count                = (pc.size + 3u) / 4u;
                params.push_back(p);
            }
            D3D12_ROOT_SIGNATURE_DESC rs = {};
            rs.NumParameters             = static_cast<UINT>(params.size());
            rs.pParameters               = params.empty() ? nullptr : params.data();
            rs.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
            ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)))
            {
                delete l;
                d->ReportError("D3D12SerializeRootSignature failed");
                return VriResult_Failure;
            }
            if (FAILED(d->Device()->CreateRootSignature(
                    0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&l->rootSig))))
            {
                delete l;
                d->ReportError("CreateRootSignature failed");
                return VriResult_Failure;
            }
            l->rootSigHash = Fnv1a(blob->GetBufferPointer(), blob->GetBufferSize());
            *out           = ToHandle(l);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout)
        {
            if (layout)
                delete PL(layout);
        }

        // Mesh-shader pipeline (MS [+ AS] + PS) via the pipeline-state-stream API
        // (ID3D12Device2). No vertex input / input assembly. Returns Success/Failure.
        VriResult CreateMeshPipeline(DeviceD3D12* d, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            PipelineLayoutD3D12*  layout = PL(desc->pipelineLayout);
            D3D12_SHADER_BYTECODE ms = {}, as = {}, ps = {};
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc&        s  = desc->shaders[i];
                const D3D12_SHADER_BYTECODE bc = {s.bytecode, s.bytecodeSize};
                if (s.stage == VriShaderStage_Mesh)
                    ms = bc;
                else if (s.stage == VriShaderStage_Task)
                    as = bc;
                else if (s.stage == VriShaderStage_Fragment)
                    ps = bc;
            }

            D3D12_RASTERIZER_DESC raster = {};
            raster.FillMode              = D3D12_FILL_MODE_SOLID;
            raster.CullMode              = ToD3DCull(desc->rasterization.cullMode);
            raster.FrontCounterClockwise =
                desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? TRUE : FALSE;
            raster.DepthClipEnable = TRUE;

            D3D12_BLEND_DESC blend               = {};
            blend.RenderTarget[0].SrcBlend       = D3D12_BLEND_ONE;
            blend.RenderTarget[0].DestBlend      = D3D12_BLEND_ZERO;
            blend.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            blend.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            blend.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            const VriColorWriteFlags      wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            blend.RenderTarget[0].RenderTargetWriteMask = static_cast<UINT8>((wm == 0 ? VriColorWrite_RGBA : wm) & 0xF);

            D3D12_DEPTH_STENCIL_DESC ds = {};
            ds.DepthEnable              = desc->depthStencil.depthTest ? TRUE : FALSE;
            ds.DepthWriteMask =
                desc->depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            ds.DepthFunc = ToD3DCompare(desc->depthStencil.depthCompareOp);
            // D3D12 has one read/write mask for both faces (VRI carries per-face); use the front's.
            ds.StencilEnable    = desc->depthStencil.stencilTest ? TRUE : FALSE;
            ds.StencilReadMask  = static_cast<UINT8>(desc->depthStencil.front.compareMask);
            ds.StencilWriteMask = static_cast<UINT8>(desc->depthStencil.front.writeMask);
            ds.FrontFace        = ToD3DStencilFace(desc->depthStencil.front);
            ds.BackFace         = ToD3DStencilFace(desc->depthStencil.back);

            D3D12_RT_FORMAT_ARRAY rtv = {};
            rtv.NumRenderTargets      = desc->outputMerger.colorNum <= 8 ? desc->outputMerger.colorNum : 8;
            for (uint32_t i = 0; i < rtv.NumRenderTargets; ++i)
                rtv.RTFormats[i] = ToDxgiFormat(desc->outputMerger.colors[i].format).format;
            DXGI_SAMPLE_DESC sample = {};
            sample.Count            = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;

            // Packed stream of subobjects (each aligned to void*); CreatePipelineState
            // reads them by their leading type tag.
            struct alignas(void*) SubRoot
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
                ID3D12RootSignature*                v;
            };
            struct alignas(void*) SubMS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubAS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubPS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubRast
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER;
                D3D12_RASTERIZER_DESC               v;
            };
            struct alignas(void*) SubBlend
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND;
                D3D12_BLEND_DESC                    v;
            };
            struct alignas(void*) SubDS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
                D3D12_DEPTH_STENCIL_DESC            v;
            };
            struct alignas(void*) SubRTV
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
                D3D12_RT_FORMAT_ARRAY               v;
            };
            struct alignas(void*) SubDSV
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
                DXGI_FORMAT                         v;
            };
            struct alignas(void*) SubSamp
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC;
                DXGI_SAMPLE_DESC                    v;
            };
            struct Stream
            {
                SubRoot  rootSig;
                SubMS    msBc;
                SubAS    asBc;
                SubPS    psBc;
                SubRast  rast;
                SubBlend blend;
                SubDS    ds;
                SubRTV   rtv;
                SubDSV   dsv;
                SubSamp  sample;
            } stream;
            stream.rootSig.v = layout->rootSig.Get();
            stream.msBc.v    = ms;
            stream.asBc.v    = as;
            stream.psBc.v    = ps;
            stream.rast.v    = raster;
            stream.blend.v   = blend;
            stream.ds.v      = ds;
            stream.rtv.v     = rtv;
            stream.dsv.v     = desc->outputMerger.depthStencilFormat != VriFormat_Unknown ?
                                   ToDxgiFormat(desc->outputMerger.depthStencilFormat).format :
                                   DXGI_FORMAT_UNKNOWN;
            stream.sample.v  = sample;

            ComPtr<ID3D12Device2> dev2;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev2))))
            {
                d->ReportError("mesh pipeline: ID3D12Device2 unavailable");
                return VriResult_Unsupported;
            }
            D3D12_PIPELINE_STATE_STREAM_DESC sdesc = {sizeof(stream), &stream};

            PipelineD3D12* p  = new PipelineD3D12 {};
            p->device         = d;
            p->rootSig        = layout->rootSig.Get();
            p->isMesh         = true;
            p->stencilEnabled = desc->depthStencil.stencilTest != VRI_FALSE;
            p->stencilRef     = desc->depthStencil.front.reference;
            if (FAILED(dev2->CreatePipelineState(&sdesc, IID_PPV_ARGS(&p->pso))))
            {
                delete p;
                d->ReportError("CreatePipelineState (mesh) failed");
                return VriResult_Failure;
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }

        // ---- pipeline-cache PSO names ----
        // ID3D12PipelineLibrary keys pipelines by name. We derive a stable name from the pipeline's
        // full logical definition (shader bytecode + root-signature hash + fixed-function state) so a
        // serialized cache reloaded next run keys the same pipeline back to the same blob.
        std::wstring HexName(uint64_t h)
        {
            wchar_t       buf[17];
            const wchar_t hex[] = L"0123456789abcdef";
            for (int i = 0; i < 16; ++i)
                buf[15 - i] = hex[(h >> (i * 4)) & 0xf];
            buf[16] = 0;
            return buf;
        }
        std::wstring GraphicsPsoName(const VriGraphicsPipelineDesc*            desc,
                                     const PipelineLayoutD3D12*                layout,
                                     const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pd)
        {
            uint64_t h = layout ? layout->rootSigHash : 0;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
                h = Fnv1a(desc->shaders[i].bytecode, desc->shaders[i].bytecodeSize, h);
            h = Fnv1a(&pd.BlendState, sizeof(pd.BlendState), h);
            h = Fnv1a(&pd.SampleMask, sizeof(pd.SampleMask), h);
            h = Fnv1a(&pd.RasterizerState, sizeof(pd.RasterizerState), h);
            h = Fnv1a(&pd.DepthStencilState, sizeof(pd.DepthStencilState), h);
            h = Fnv1a(&pd.IBStripCutValue, sizeof(pd.IBStripCutValue), h);
            h = Fnv1a(&pd.PrimitiveTopologyType, sizeof(pd.PrimitiveTopologyType), h);
            h = Fnv1a(&pd.NumRenderTargets, sizeof(pd.NumRenderTargets), h);
            h = Fnv1a(pd.RTVFormats, sizeof(pd.RTVFormats), h);
            h = Fnv1a(&pd.DSVFormat, sizeof(pd.DSVFormat), h);
            h = Fnv1a(&pd.SampleDesc, sizeof(pd.SampleDesc), h);
            for (UINT i = 0; i < pd.InputLayout.NumElements; ++i)
            {
                const D3D12_INPUT_ELEMENT_DESC& e = pd.InputLayout.pInputElementDescs[i];
                h = Fnv1a(e.SemanticName, std::strlen(e.SemanticName) + 1, h); // pointer -> hash content
                h = Fnv1a(&e.SemanticIndex, sizeof(e.SemanticIndex), h);
                h = Fnv1a(&e.Format, sizeof(e.Format), h);
                h = Fnv1a(&e.InputSlot, sizeof(e.InputSlot), h);
                h = Fnv1a(&e.AlignedByteOffset, sizeof(e.AlignedByteOffset), h);
                h = Fnv1a(&e.InputSlotClass, sizeof(e.InputSlotClass), h);
                h = Fnv1a(&e.InstanceDataStepRate, sizeof(e.InstanceDataStepRate), h);
            }
            return HexName(h);
        }

        // One packed pipeline-state-stream subobject: a type tag followed by its payload, aligned to
        // void*. Used to build a graphics PSO with a ViewInstancing subobject (multiview), which the
        // classic D3D12_GRAPHICS_PIPELINE_STATE_DESC can't express.
        template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename T>
        struct alignas(void*) PsoSub
        {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
            T                                   value {};
        };

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*                     device,
                                                  const VriGraphicsPipelineDesc* desc,
                                                  VriPipeline**                  out)
        {
            DeviceD3D12*         d      = Dev(device);
            PipelineLayoutD3D12* layout = PL(desc->pipelineLayout);

            for (uint32_t i = 0; i < desc->shaderNum; ++i)
                if (desc->shaders[i].stage == VriShaderStage_Mesh || desc->shaders[i].stage == VriShaderStage_Task)
                    return CreateMeshPipeline(d, desc, out);

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature                     = layout->rootSig.Get();
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc&        s  = desc->shaders[i];
                const D3D12_SHADER_BYTECODE bc = {s.bytecode, s.bytecodeSize};
                if (s.stage == VriShaderStage_Vertex)
                    pd.VS = bc;
                else if (s.stage == VriShaderStage_Fragment)
                    pd.PS = bc;
                else if (s.stage == VriShaderStage_Geometry)
                    pd.GS = bc;
                else if (s.stage == VriShaderStage_TessControl)
                    pd.HS = bc;
                else if (s.stage == VriShaderStage_TessEval)
                    pd.DS = bc;
            }

            // Input layout (per-attribute). VRI attributes are location-indexed, but D3D
            // matches vertex inputs by semantic name+index. Reflect the VS input signature
            // (register N == VRI attribute/location N) so the layout uses the real
            // semantics (POSITION, COLOR0, ...) Slang emitted - no hardcoded convention.
            struct SemInfo
            {
                std::string name;
                UINT        index = 0;
            };
            std::vector<SemInfo> byRegister; // owns the semantic strings the layout points at
            if (pd.VS.pShaderBytecode && desc->vertexInput.attributeNum)
            {
                ComPtr<ID3D12ShaderReflection> refl;
                if (SUCCEEDED(D3DReflect(pd.VS.pShaderBytecode, pd.VS.BytecodeLength, IID_PPV_ARGS(&refl))))
                {
                    D3D12_SHADER_DESC sd = {};
                    refl->GetDesc(&sd);
                    byRegister.resize(sd.InputParameters);
                    for (UINT i = 0; i < sd.InputParameters; ++i)
                    {
                        D3D12_SIGNATURE_PARAMETER_DESC p = {};
                        refl->GetInputParameterDesc(i, &p);
                        if (p.Register < byRegister.size())
                            byRegister[p.Register] = {p.SemanticName ? p.SemanticName : "TEXCOORD", p.SemanticIndex};
                    }
                }
            }
            std::vector<D3D12_INPUT_ELEMENT_DESC> elems;
            std::vector<PipelineGraphicsVB>       vbStrides;
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc& a    = desc->vertexInput.attributes[i];
                uint32_t                      slot = a.streamIndex, stride = 0;
                bool                          perInstance = false;
                if (a.streamIndex < desc->vertexInput.streamNum)
                {
                    slot        = desc->vertexInput.streams[a.streamIndex].bindingSlot;
                    stride      = static_cast<uint32_t>(desc->vertexInput.streams[a.streamIndex].stride);
                    perInstance = desc->vertexInput.streams[a.streamIndex].stepRate == VriVertexStepRate_PerInstance;
                }
                D3D12_INPUT_ELEMENT_DESC e = {};
                if (i < byRegister.size() && !byRegister[i].name.empty())
                {
                    e.SemanticName  = byRegister[i].name.c_str();
                    e.SemanticIndex = byRegister[i].index;
                }
                else
                {
                    e.SemanticName  = "TEXCOORD";
                    e.SemanticIndex = i;
                }
                e.Format               = ToDxgiVertexFormat(a.format);
                e.InputSlot            = slot;
                e.AlignedByteOffset    = a.offset;
                e.InputSlotClass       = perInstance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA :
                                                       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
                e.InstanceDataStepRate = perInstance ? 1u : 0u;
                elems.push_back(e);
                bool have = false;
                for (auto& vb : vbStrides)
                    if (vb.slot == slot)
                    {
                        have = true;
                        break;
                    }
                if (!have)
                    vbStrides.push_back({stride, slot});
            }
            if (!elems.empty())
            {
                pd.InputLayout.pInputElementDescs = elems.data();
                pd.InputLayout.NumElements        = static_cast<UINT>(elems.size());
            }

            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = ToD3DCull(desc->rasterization.cullMode);
            pd.RasterizerState.FrontCounterClockwise =
                desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? TRUE : FALSE;
            pd.RasterizerState.DepthClipEnable    = TRUE;
            pd.RasterizerState.ConservativeRaster = desc->rasterization.conservativeRaster ?
                                                        D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON :
                                                        D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            pd.BlendState.RenderTarget[0].SrcBlend       = D3D12_BLEND_ONE;
            pd.BlendState.RenderTarget[0].DestBlend      = D3D12_BLEND_ZERO;
            pd.BlendState.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            pd.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            pd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            pd.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            if (c0 && c0->blend.enable)
            {
                pd.BlendState.RenderTarget[0].BlendEnable    = TRUE;
                pd.BlendState.RenderTarget[0].SrcBlend       = ToD3DBlend(c0->blend.srcColor);
                pd.BlendState.RenderTarget[0].DestBlend      = ToD3DBlend(c0->blend.dstColor);
                pd.BlendState.RenderTarget[0].BlendOp        = ToD3DBlendOp(c0->blend.colorOp);
                pd.BlendState.RenderTarget[0].SrcBlendAlpha  = ToD3DBlend(c0->blend.srcAlpha);
                pd.BlendState.RenderTarget[0].DestBlendAlpha = ToD3DBlend(c0->blend.dstAlpha);
                pd.BlendState.RenderTarget[0].BlendOpAlpha   = ToD3DBlendOp(c0->blend.alphaOp);
            }
            const VriColorWriteFlags wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            pd.BlendState.RenderTarget[0].RenderTargetWriteMask =
                static_cast<UINT8>((wm == 0 ? VriColorWrite_RGBA : wm) & 0xF);

            pd.DepthStencilState.DepthEnable = desc->depthStencil.depthTest ? TRUE : FALSE;
            pd.DepthStencilState.DepthWriteMask =
                desc->depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            pd.DepthStencilState.DepthFunc = ToD3DCompare(desc->depthStencil.depthCompareOp);
            // D3D12 has one read/write mask for both faces (VRI carries per-face); use the front's.
            // The stencil reference is dynamic state in D3D12 (OMSetStencilRef), set when the pipeline binds.
            pd.DepthStencilState.StencilEnable    = desc->depthStencil.stencilTest ? TRUE : FALSE;
            pd.DepthStencilState.StencilReadMask  = static_cast<UINT8>(desc->depthStencil.front.compareMask);
            pd.DepthStencilState.StencilWriteMask = static_cast<UINT8>(desc->depthStencil.front.writeMask);
            pd.DepthStencilState.FrontFace        = ToD3DStencilFace(desc->depthStencil.front);
            pd.DepthStencilState.BackFace         = ToD3DStencilFace(desc->depthStencil.back);

            pd.SampleMask            = UINT_MAX;
            pd.PrimitiveTopologyType = ToD3DTopologyType(desc->inputAssembly.topology);
            pd.NumRenderTargets      = desc->outputMerger.colorNum <= 8 ? desc->outputMerger.colorNum : 8;
            for (uint32_t i = 0; i < pd.NumRenderTargets; ++i)
                pd.RTVFormats[i] = ToDxgiFormat(desc->outputMerger.colors[i].format).format;
            pd.DSVFormat        = desc->outputMerger.depthStencilFormat != VriFormat_Unknown ?
                                      ToDxgiFormat(desc->outputMerger.depthStencilFormat).format :
                                      DXGI_FORMAT_UNKNOWN;
            pd.SampleDesc.Count = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;

            PipelineD3D12* p  = new PipelineD3D12 {};
            p->device         = d;
            p->rootSig        = layout->rootSig.Get();
            p->vbStrides      = std::move(vbStrides);
            p->topology       = ToD3DTopology(desc->inputAssembly.topology, desc->tessellation.patchControlPoints);
            p->stencilEnabled = desc->depthStencil.stencilTest != VRI_FALSE;
            p->stencilRef     = desc->depthStencil.front.reference;

            // Multiview: D3D12 ViewInstancing fans one draw out to N view instances, each writing its
            // own render-target array slice while the shader reads SV_ViewID (single-pass stereo).
            // The classic PSO desc can't carry ViewInstancing, so build a pipeline-state stream from
            // the same state. (Pipeline cache is skipped on this path.)
            if (desc->outputMerger.viewMask != 0)
            {
                ComPtr<ID3D12Device2> dev2;
                if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev2))))
                {
                    delete p;
                    d->ReportError("multiview: ID3D12Device2 unavailable");
                    return VriResult_Unsupported;
                }
                D3D12_VIEW_INSTANCE_LOCATION locs[4]   = {};
                uint32_t                     viewCount = 0;
                for (uint32_t bit = 0; bit < 32 && viewCount < 4; ++bit)
                    if (desc->outputMerger.viewMask & (1u << bit))
                    {
                        locs[viewCount].RenderTargetArrayIndex = viewCount; // view N -> array slice N
                        ++viewCount;
                    }
                D3D12_VIEW_INSTANCING_DESC vi = {};
                vi.ViewInstanceCount          = viewCount;
                vi.pViewInstanceLocations     = locs;

                D3D12_RT_FORMAT_ARRAY rtfmt = {};
                rtfmt.NumRenderTargets      = pd.NumRenderTargets;
                for (UINT i = 0; i < pd.NumRenderTargets; ++i)
                    rtfmt.RTFormats[i] = pd.RTVFormats[i];

                struct Stream
                {
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>  rootSig;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>             vs;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>             ps;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, D3D12_SHADER_BYTECODE>             gs;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS, D3D12_SHADER_BYTECODE>             hs;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS, D3D12_SHADER_BYTECODE>             ds;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, D3D12_INPUT_LAYOUT_DESC> inputLayout;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topo;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>                 rast;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC>                           blend;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC>           depth;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT>                            sampleMask;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtv;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT>            dsv;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC>                sample;
                    PsoSub<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, D3D12_VIEW_INSTANCING_DESC>  viewInst;
                } stream;
                stream.rootSig.value     = pd.pRootSignature;
                stream.vs.value          = pd.VS;
                stream.ps.value          = pd.PS;
                stream.gs.value          = pd.GS;
                stream.hs.value          = pd.HS;
                stream.ds.value          = pd.DS;
                stream.inputLayout.value = pd.InputLayout;
                stream.topo.value        = pd.PrimitiveTopologyType;
                stream.rast.value        = pd.RasterizerState;
                stream.blend.value       = pd.BlendState;
                stream.depth.value       = pd.DepthStencilState;
                stream.sampleMask.value  = pd.SampleMask;
                stream.rtv.value         = rtfmt;
                stream.dsv.value         = pd.DSVFormat;
                stream.sample.value      = pd.SampleDesc;
                stream.viewInst.value    = vi;

                D3D12_PIPELINE_STATE_STREAM_DESC sdesc = {sizeof(stream), &stream};
                if (FAILED(dev2->CreatePipelineState(&sdesc, IID_PPV_ARGS(&p->pso))))
                {
                    delete p;
                    d->ReportError("CreatePipelineState (multiview) failed");
                    return VriResult_Failure;
                }
                *out = ToHandle(p);
                return VriResult_Success;
            }

            // Pipeline cache: load a matching PSO from the library, else create it and store it.
            PipelineCacheD3D12* cache = desc->pipelineCache ? PipeCache(desc->pipelineCache) : nullptr;
            HRESULT             hr    = E_FAIL;
            std::wstring        name;
            if (cache && cache->lib)
            {
                name = GraphicsPsoName(desc, layout, pd);
                hr   = cache->lib->LoadGraphicsPipeline(name.c_str(), &pd, IID_PPV_ARGS(&p->pso));
            }
            if (FAILED(hr))
            {
                if (FAILED(d->Device()->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&p->pso))))
                {
                    delete p;
                    d->ReportError("CreateGraphicsPipelineState failed");
                    return VriResult_Failure;
                }
                if (cache && cache->lib)
                    cache->lib->StorePipeline(name.c_str(), p->pso.Get()); // E_INVALIDARG = already stored
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*                    device,
                                                 const VriComputePipelineDesc* desc,
                                                 VriPipeline**                 out)
        {
            DeviceD3D12*                      d      = Dev(device);
            PipelineLayoutD3D12*              layout = PL(desc->pipelineLayout);
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd     = {};
            pd.pRootSignature                        = layout->rootSig.Get();
            pd.CS                                    = {desc->shader.bytecode, desc->shader.bytecodeSize};
            PipelineD3D12* p                         = new PipelineD3D12 {};
            p->device                                = d;
            p->rootSig                               = layout->rootSig.Get();
            p->isCompute                             = true;

            PipelineCacheD3D12* cache = desc->pipelineCache ? PipeCache(desc->pipelineCache) : nullptr;
            HRESULT             hr    = E_FAIL;
            std::wstring        name;
            if (cache && cache->lib)
            {
                uint64_t h = layout ? layout->rootSigHash : 0;
                h          = Fnv1a(desc->shader.bytecode, desc->shader.bytecodeSize, h);
                name       = HexName(h);
                hr         = cache->lib->LoadComputePipeline(name.c_str(), &pd, IID_PPV_ARGS(&p->pso));
            }
            if (FAILED(hr))
            {
                if (FAILED(d->Device()->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p->pso))))
                {
                    delete p;
                    d->ReportError("CreateComputePipelineState failed");
                    return VriResult_Failure;
                }
                if (cache && cache->lib)
                    cache->lib->StorePipeline(name.c_str(), p->pso.Get());
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (pipeline)
                delete Pipe(pipeline);
        }
        VriResult VRI_CALL CreateDescriptorPool(VriDevice*                   device,
                                                const VriDescriptorPoolDesc* desc,
                                                VriDescriptorPool**          out)
        {
            DeviceD3D12*         d = Dev(device);
            DescriptorPoolD3D12* p = new DescriptorPoolD3D12 {};
            p->device              = d;
            // Shader-visible heaps for SRV/CBV/UAV + samplers (descriptor tables bind from
            // these). Sized from the pool request with sane minimums; root CBVs use neither.
            const uint32_t srvCap = (desc->textureMaxNum + desc->structuredBufferMaxNum + desc->storageBufferMaxNum +
                                     desc->storageTextureMaxNum + 16);
            const uint32_t samplerCap     = (desc->samplerMaxNum + 16);
            D3D12_DESCRIPTOR_HEAP_DESC hd = {};
            hd.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors             = srvCap;
            hd.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (FAILED(d->Device()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p->srvHeap))))
            {
                delete p;
                d->ReportError("CreateDescriptorHeap (SRV) failed");
                return VriResult_Failure;
            }
            hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            hd.NumDescriptors = samplerCap;
            if (FAILED(d->Device()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p->samplerHeap))))
            {
                delete p;
                d->ReportError("CreateDescriptorHeap (sampler) failed");
                return VriResult_Failure;
            }
            p->srvSize     = d->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            p->samplerSize = d->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            *out           = ToHandle(p);
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            DescriptorPoolD3D12* p = DPool(pool);
            p->srvNext             = 0;
            p->samplerNext         = 0;
        }
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (pool)
                delete DPool(pool);
        }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*       pool,
                                                  const VriPipelineLayout* layout,
                                                  uint32_t                 setIndex,
                                                  VriDescriptorSet**       outSets,
                                                  uint32_t                 setNum)
        {
            DescriptorPoolD3D12*       p  = DPool(pool);
            const PipelineLayoutD3D12* l  = PL(layout);
            const LayoutSetD3D12*      si = l->FindSet(setIndex);
            for (uint32_t i = 0; i < setNum; ++i)
            {
                DescriptorSetD3D12* s = new DescriptorSetD3D12 {};
                s->device             = p->device;
                s->pool               = p;
                s->layout             = l;
                s->setIndex           = setIndex;
                if (si && si->srvCount > 0) // sub-allocate this set's SRV block
                {
                    s->srvCpu = p->srvHeap->GetCPUDescriptorHandleForHeapStart();
                    s->srvCpu.ptr += static_cast<SIZE_T>(p->srvNext) * p->srvSize;
                    s->srvGpu = p->srvHeap->GetGPUDescriptorHandleForHeapStart();
                    s->srvGpu.ptr += static_cast<UINT64>(p->srvNext) * p->srvSize;
                    p->srvNext += si->srvCount;
                }
                if (si && si->samplerCount > 0)
                {
                    s->samplerCpu = p->samplerHeap->GetCPUDescriptorHandleForHeapStart();
                    s->samplerCpu.ptr += static_cast<SIZE_T>(p->samplerNext) * p->samplerSize;
                    s->samplerGpu = p->samplerHeap->GetGPUDescriptorHandleForHeapStart();
                    s->samplerGpu.ptr += static_cast<UINT64>(p->samplerNext) * p->samplerSize;
                    p->samplerNext += si->samplerCount;
                }
                outSets[i] = ToHandle(s);
            }
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*                   set,
                                             uint32_t                            baseRange,
                                             uint32_t                            rangeNum,
                                             const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetD3D12*                    s = DSet(set);
            DeviceD3D12*                           d = s->device;
            std::vector<const LayoutBindingD3D12*> setBindings; // this set's bindings in declaration order
            for (const LayoutBindingD3D12& b : s->layout->bindings)
                if (b.set == s->setIndex)
                    setBindings.push_back(&b);
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const uint32_t idx = baseRange + r;
                if (idx >= setBindings.size() || updates[r].descriptorNum == 0)
                    continue;
                const LayoutBindingD3D12* lb = setBindings[idx];
                s->bound[lb->binding] =
                    reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[0]); // root CBVs bind [0] at draw
                if (lb->type == VriDescriptorType_ConstantBuffer)
                    continue;
                // A range may carry an array of descriptors (descriptor indexing / bindless):
                // write each into consecutive heap slots from heapOffset + baseDescriptor.
                for (uint32_t k = 0; k < updates[r].descriptorNum; ++k)
                {
                    const DescriptorD3D12* v = reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[k]);
                    if (!v)
                        continue;
                    const SIZE_T slot = static_cast<SIZE_T>(lb->heapOffset) + updates[r].baseDescriptor + k;
                    if (lb->type == VriDescriptorType_Sampler) // realize into the set's sampler heap slot
                    {
                        D3D12_SAMPLER_DESC          sd = ToD3DSampler(v->sampler);
                        D3D12_CPU_DESCRIPTOR_HANDLE h  = s->samplerCpu;
                        h.ptr += slot * s->pool->samplerSize;
                        d->Device()->CreateSampler(&sd, h);
                    }
                    else if (lb->type == VriDescriptorType_AccelerationStructure) // TLAS SRV (no resource, by GPU VA)
                    {
                        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
                        srv.ViewDimension                   = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                        srv.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        srv.RaytracingAccelerationStructure.Location = v->accelAddress;
                        D3D12_CPU_DESCRIPTOR_HANDLE h                = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateShaderResourceView(nullptr, &srv, h);
                    }
                    else if (lb->type == VriDescriptorType_StorageTexture && v->texture) // UAV
                    {
                        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
                        uav.Format                           = v->texture->format;
                        uav.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE2D;
                        D3D12_CPU_DESCRIPTOR_HANDLE h        = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateUnorderedAccessView(v->texture->resource.Get(), nullptr, &uav, h);
                    }
                    else if (v->texture) // SRV (sampled texture) into the set's CBV/SRV/UAV heap slot
                    {
                        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
                        srv.Format =
                            v->texture->srvFormat; // typed sampleable format (R32_FLOAT for a sampled D32 depth)
                        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        const bool isCube =
                            v->viewType == VriTextureViewType_Cube || v->viewType == VriTextureViewType_CubeArray;
                        if (isCube) // a TextureCube SRV samples a 6-layer resource by direction
                        {
                            srv.ViewDimension         = D3D12_SRV_DIMENSION_TEXTURECUBE;
                            srv.TextureCube.MipLevels = v->texture->mipNum;
                        }
                        else if (v->texture->layerNum > 1)
                        {
                            srv.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                            srv.Texture2DArray.MipLevels = v->texture->mipNum;
                            srv.Texture2DArray.ArraySize = v->texture->layerNum;
                        }
                        else
                        {
                            srv.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                            srv.Texture2D.MipLevels = v->texture->mipNum;
                        }
                        D3D12_CPU_DESCRIPTOR_HANDLE h = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateShaderResourceView(v->texture->resource.Get(), &srv, h);
                    }
                }
            }
        }
