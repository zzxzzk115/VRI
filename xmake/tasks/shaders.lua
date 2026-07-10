-- `xmake shaders` - compile the test .slang shaders to embedded SPIR-V headers
-- using the in-process vri-shaderc tool (which links the Slang compiler library).
-- The headers are committed and #included by the tests/examples.

task("shaders")
    set_menu {
        usage = "xmake shaders",
        description = "Compile test .slang shaders to embedded SPIR-V headers via vri-shaderc.",
        options = {}
    }
    on_run(function ()
        -- snake_case file stem -> g_camelCaseSpv
        local function camel(name)
            local parts = name:split("_", {plain = true})
            local s = ""
            for i, p in ipairs(parts) do
                s = (i == 1) and p or (s .. p:sub(1, 1):upper() .. p:sub(2))
            end
            return s
        end

        -- make sure the tool is built, then locate the freshest executable
        os.execv(os.programfile(), {"build", "vri-shaderc"})
        local ext = is_host("windows") and ".exe" or ""
        local matches = os.files(path.join(os.projectdir(), "build", "**", "vri-shaderc" .. ext))
        assert(#matches > 0, "vri-shaderc not built - enable the vri_build_tools option")
        local exe = matches[1]
        for _, m in ipairs(matches) do
            if os.mtime(m) > os.mtime(exe) then exe = m end
        end

        -- All shaders live under shaders/, split by owner: lib (library-internal, e.g. the built-in
        -- ImGui renderer), examples (example-only), tests (test-only), and common (shared by tests +
        -- examples). Each header is emitted next to its .slang.
        local dirs = {
            path.join(os.projectdir(), "shaders", "lib"),
            path.join(os.projectdir(), "shaders", "examples"),
            path.join(os.projectdir(), "shaders", "tests"),
            path.join(os.projectdir(), "shaders", "common"),
        }
        local slangs = {}
        for _, d in ipairs(dirs) do
            for _, s in ipairs(os.files(path.join(d, "*.slang"))) do
                table.insert(slangs, s)
            end
        end
        if #slangs == 0 then
            cprint("${yellow}[shaders]${clear} no .slang files found")
            return
        end

        local stem = camel
        for _, slang in ipairs(slangs) do
            local base = path.basename(slang)
            local dir  = path.directory(slang) -- emit each header beside its .slang
            -- SPIR-V (Vulkan) + WGSL (WebGPU) from the same Slang source
            local spvHeader = path.join(dir, base .. "_spv.h")
            local spvVar = "g_" .. stem(base) .. "Spv"
            os.runv(exe, {slang, "-o", spvHeader, "--var", spvVar, "--target", "spirv"})

            -- WGSL is best-effort: WebGPU has no geometry/tessellation stages, so
            -- those shaders only produce SPIR-V (consumed by Vulkan + desktop GL).
            local wgslHeader = path.join(dir, base .. "_wgsl.h")
            local wgslVar = "g_" .. stem(base) .. "Wgsl"
            local ok = try { function() os.runv(exe, {slang, "-o", wgslHeader, "--var", wgslVar, "--target", "wgsl"}) return true end }
            if ok then
                cprint("${green}[shaders]${clear} %s -> %s + %s", path.filename(slang), path.filename(spvHeader), path.filename(wgslHeader))
            else
                cprint("${yellow}[shaders]${clear} %s -> %s (no WGSL: unsupported stage)", path.filename(slang), path.filename(spvHeader))
            end

            -- DXBC (Direct3D 12) is Windows-host only (Slang shells to fxc) and best-effort.
            -- Emits per-stage byte arrays g_<name>DxbcVS/PS/... (see vri-shaderc --target dxbc).
            if is_host("windows") then
                local dxbcHeader = path.join(dir, base .. "_dxbc.h")
                local dxbcVar = "g_" .. stem(base) .. "Dxbc"
                local dok = try { function() os.runv(exe, {slang, "-o", dxbcHeader, "--var", dxbcVar, "--target", "dxbc"}) return true end }
                if dok then
                    cprint("${green}[shaders]${clear}   + %s", path.filename(dxbcHeader))
                else
                    cprint("${yellow}[shaders]${clear}   (no DXBC for %s: unsupported stage)", path.filename(slang))
                end

                -- DXIL (SM6.x) for shaders D3D12 needs as DXIL: mesh shaders (sm_6_5)
                -- and DXR ray tracing (lib_6_3, one library w/ all RT entry points).
                -- DXBC (FXC, sm_5_1) cannot express these stages, hence DXIL.
                -- multiview_vert reads SV_ViewID (D3D12 ViewInstancing) which needs SM6.1; the paired
                -- fragment carries no ViewID so plain sm_6_0 suffices.
                local dxilProfiles = { triangle_mesh = "sm_6_5", rt_triangle = "lib_6_3", rt_gltf = "lib_6_5", rt_gltf_rayquery = "cs_6_5", rt_gltf_software = "cs_6_5", rt_pathtrace = "cs_6_5", rt_pathtrace_software = "cs_6_5", mesh_frag = "sm_6_0", rayquery = "cs_6_5", rt_rayquery = "cs_6_5", barycentric = "sm_6_1", barywire = "sm_6_1", wave = "cs_6_0", wave_vis = "cs_6_0", multiview_vert = "sm_6_1", multiview_frag = "sm_6_0" }
                local prof = dxilProfiles[base]
                if prof then
                    local dxilHeader = path.join(dir, base .. "_dxil.h")
                    local dxilVar = "g_" .. stem(base) .. "Dxil"
                    local xok = try { function() os.runv(exe, {slang, "-o", dxilHeader, "--var", dxilVar, "--target", "dxil", "--profile", prof}) return true end }
                    if xok then
                        cprint("${green}[shaders]${clear}   + %s (%s)", path.filename(dxilHeader), prof)
                    else
                        cprint("${yellow}[shaders]${clear}   (no DXIL for %s)", path.filename(slang))
                    end
                end
            end
        end
    end)
task_end()
