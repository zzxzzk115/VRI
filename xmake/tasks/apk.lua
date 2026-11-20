-- `xmake apk example-cube` - build an example for the configured Android ABI and package it as a
-- debug APK. The Gradle project (examples/android) is staged under build/android/<target> with the
-- example's library and its vri.android_assets, then Gradle's assembleDebug signs it with the
-- Android SDK's debug key. Needs Gradle and a JDK; the SDK comes from --sdk, `xmake f --android_sdk`
-- or ANDROID_HOME.

task("apk")
    set_menu {
        usage = "xmake apk [--gradle=PATH] [--sdk=DIR] <target>",
        description = "Package an Android example as a debug APK.",
        options = {
            {nil, "gradle", "kv", nil, "Gradle executable. Defaults to gradle on PATH."},
            {nil, "sdk", "kv", nil, "Android SDK directory. Defaults to --android_sdk, then ANDROID_HOME."},
            {nil, "target", "v", nil, "The example target, e.g. example-cube."}
        }
    }
    on_run(function ()
        import("core.base.option")
        import("core.base.task")
        import("core.project.config")
        import("core.project.project")
        import("lib.detect.find_program")

        config.load()
        if config.get("plat") ~= "android" then
            raise("xmake apk needs an Android configuration: xmake f -p android -a x86_64 --vri_build_examples=y ...")
        end
        local name = option.get("target")
        if not name then
            raise("usage: xmake apk <target>")
        end
        local sdk = option.get("sdk") or config.get("android_sdk") or os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT")
        if not sdk or not os.isdir(sdk) then
            raise("Android SDK not found: pass --sdk=DIR, configure --android_sdk or set ANDROID_HOME")
        end
        local gradle = option.get("gradle") or find_program("gradle")
        if not gradle then
            raise("Gradle not found: pass --gradle=PATH or put gradle on PATH")
        end

        task.run("build", {target = name})
        local target = project.target(name)
        if not target or target:kind() ~= "shared" then
            raise("%s is not an Android example (see the vri.android_example rule in xmake/examples.lua)", name)
        end

        -- Regenerated from scratch on every run: it holds nothing but copies and Gradle's outputs.
        local stage = path.join(config.builddir(), "android", name)
        os.tryrm(stage)
        local template = path.join(os.projectdir(), "examples", "android")
        for _, file in ipairs({"build.gradle", "settings.gradle", "gradle.properties"}) do
            os.cp(path.join(template, file), path.join(stage, file))
        end
        local library = path.basename(target:targetfile()):gsub("^lib", "")
        local manifest = io.readfile(path.join(template, "AndroidManifest.xml"))
        manifest = manifest:gsub("@VRI_LIBRARY@", library):gsub("@VRI_LABEL@", "VRI " .. name:gsub("^example%-", ""))
        io.writefile(path.join(stage, "AndroidManifest.xml"), manifest)
        io.writefile(path.join(stage, "local.properties"), "sdk.dir=" .. path.absolute(sdk):gsub("\\", "/") .. "\n")
        os.cp(target:targetfile(), path.join(stage, "jniLibs", config.arch(), path.filename(target:targetfile())))
        os.mkdir(path.join(stage, "assets"))
        -- A file lands at the assets root, a directory under its own name.
        for _, asset in ipairs(table.wrap(target:values("vri.android_assets"))) do
            os.cp(asset, path.join(stage, "assets", path.filename(asset)))
        end

        local id = "org.vri.example." .. name:gsub("^example%-", "")
        os.execv(gradle, {"--project-dir", stage, "-PvriApplicationId=" .. id, "assembleDebug"})
        local apks = os.files(path.join(stage, "build", "outputs", "apk", "debug", "*.apk"))
        if #apks ~= 1 then
            raise("expected one debug APK under %s", path.join(stage, "build", "outputs", "apk", "debug"))
        end
        local apk = path.join(config.builddir(), "android", name .. ".apk")
        os.cp(apks[1], apk)
        cprint("${bright green}apk:${clear} %s (%s)", path.absolute(apk), id)
    end)
task_end()
