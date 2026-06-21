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

        local dir = path.join(os.projectdir(), "tests", "shaders")
        local slangs = os.files(path.join(dir, "*.slang"))
        if #slangs == 0 then
            cprint("${yellow}[shaders]${clear} no .slang files under %s", dir)
            return
        end

        local stem = camel
        for _, slang in ipairs(slangs) do
            local base = path.basename(slang)
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
            end
        end
    end)
task_end()
