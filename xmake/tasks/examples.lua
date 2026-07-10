-- `xmake examples` - run all desktop examples, optionally bounded by a fixed frame count.
-- By default this also scans VRI validation output and fails on errors/non-performance warnings.

task("examples")
    set_menu {
        usage = "xmake examples [--backend=vulkan,d3d12,opengl] [--frames=N] [--no-strict]",
        description = "Run all examples with VRI_MAX_FRAMES and fail on validation errors/warnings.",
        options = {
            {nil, "backend", "kv", nil, "Comma-separated backend list. Defaults to desktop backends for this host; Metal is skipped by default."},
            {nil, "frames", "kv", nil, "Frames to present per example. Omit to run until the example exits."},
            {nil, "no-strict", "k", nil, "Do not fail on [VRI][ERROR] or non-performance [VRI][WARN] output."}
        }
    }
    on_run(function ()
        import("core.base.option")
        import("core.project.config")

        config.load()

        local function load_examples()
            local text = io.readfile(path.join(os.projectdir(), "xmake", "examples.lua"))
            local out = {}
            for entry in text:gmatch("{%s*name%s*=%s*\"[^\"]+\".-}") do
                local name = entry:match("name%s*=%s*\"([^\"]+)\"")
                local target = entry:match("target%s*=%s*\"([^\"]+)\"") or ("example-" .. name)
                local desktop_only = entry:find("desktop_only%s*=%s*true") ~= nil
                -- Opt-in examples are only built when their config option is on (mirrors
                -- _vri_example_enabled in xmake/examples.lua) - skip them here too, or the
                -- run gate would fail on a binary the build legitimately never produced.
                local enabled = true
                if entry:find("requires_cuda%s*=%s*true") and not config.get("vri_build_cuda_interop") then
                    enabled = false
                end
                if entry:find("requires_openxr%s*=%s*true") and not config.get("vri_build_openxr") then
                    enabled = false
                end
                if name and enabled and not (desktop_only and is_plat("wasm")) then
                    table.insert(out, target)
                end
            end
            return out
        end

        local examples = load_examples()
        assert(#examples > 0, "no examples registered; check examples/xmake.lua")
        local function split_csv(s)
            local out = {}
            for item in tostring(s):gmatch("[^,]+") do
                item = item:trim()
                if #item > 0 then table.insert(out, item) end
            end
            return out
        end

        local backend_opt = option.get("backend")
        local backends
        if backend_opt then
            backends = split_csv(backend_opt)
        elseif is_host("windows") then
            backends = {"vulkan", "d3d12", "opengl"}
        elseif is_host("linux") then
            backends = {"vulkan", "opengl"}
        elseif is_host("macosx") then
            -- Metal is intentionally skipped by default for now; pass --backend=metal to include it.
            backends = {"vulkan", "opengl"}
        else
            backends = {"auto"}
        end

        local frames = option.get("frames")
        if frames then
            frames = tostring(frames)
        end
        local strict = not option.get("no-strict")
        local failures = {}

        local function append_failure(msg)
            table.insert(failures, msg)
            cprint("${red}%s${clear}", msg)
        end

        local function psquote(s)
            return "'" .. tostring(s):gsub("'", "''") .. "'"
        end

        local function shquote(s)
            return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
        end

        local function target_file(targetname)
            local ext = is_host("windows") and ".exe" or ""
            local matches = os.files(path.join(os.projectdir(), "build", "**", targetname .. ext))
            if #matches == 0 then
                append_failure(string.format("%s: binary not found under build/**; run `xmake build -y %s` first", targetname, targetname))
                return nil
            end
            local file = matches[1]
            for _, candidate in ipairs(matches) do
                if os.mtime(candidate) > os.mtime(file) then
                    file = candidate
                end
            end
            file = path.absolute(file)
            return file, path.directory(file)
        end
        local function run_capture(targetname, backend)
            local file, rundir = target_file(targetname)
            if not file then
                return "__VRI_EXIT__-1\n"
            end
            if is_host("windows") then
                local frameenv = frames and string.format("$env:VRI_MAX_FRAMES=%s; ", psquote(frames)) or ""
                local cmd = string.format("Push-Location %s; $env:VRI_API=%s; %s& %s *>&1; $code=$LASTEXITCODE; Pop-Location; Write-Output ('__VRI_EXIT__' + $code); exit 0",
                    psquote(rundir), psquote(backend), frameenv, psquote(file))
                return os.iorunv("powershell", {"-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd})
            else
                local frameenv = frames and string.format("VRI_MAX_FRAMES=%s ", shquote(frames)) or ""
                local cmd = string.format("cd %s && VRI_API=%s %s%s 2>&1; code=$?; echo __VRI_EXIT__$code; exit 0",
                    shquote(rundir), shquote(backend), frameenv, shquote(file))
                return os.iorunv("sh", {"-c", cmd})
            end
        end
        for _, backend in ipairs(backends) do
            for _, target in ipairs(examples) do
                cprint("${bright}examples:${clear} %s / %s", backend, target)
                local output = run_capture(target, backend)
                local exitcode = nil
                for line in output:gmatch("[^\r\n]+") do
                    local marker = line:match("__VRI_EXIT__(%-?%d+)")
                    if marker then
                        exitcode = tonumber(marker)
                    else
                        print(line)
                        if strict then
                            local is_error = line:find("[VRI][ERROR]", 1, true) ~= nil or
                                             line:lower():find("validation error", 1, true) ~= nil
                            local is_vri_warn = line:find("[VRI][WARN]", 1, true) ~= nil
                            local is_perf = line:lower():find("performance", 1, true) ~= nil
                            if is_error or (is_vri_warn and not is_perf) then
                                append_failure(string.format("%s / %s: %s", backend, target, line))
                            end
                        end
                    end
                end
                if exitcode ~= 0 then
                    append_failure(string.format("%s / %s: exited with %s", backend, target, tostring(exitcode or "unknown")))
                end
            end
        end

        if #failures > 0 then
            raise("examples task failed with %d issue(s)", #failures)
        end
        local frame_desc = frames and (frames .. " frame(s)") or "unbounded"
        cprint("${green}examples:${clear} all runs passed (%d example targets x %d backend(s), %s)", #examples, #backends, frame_desc)
    end)
task_end()
