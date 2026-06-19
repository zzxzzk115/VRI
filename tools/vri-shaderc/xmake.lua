-- vri-shaderc: offline Slang -> SPIR-V compiler that links the Slang compiler
-- library (drives it in-process), rather than shelling out to slangc. Uses the
-- prebuilt Slang bundled with the Vulkan SDK via the `slangsdk` rule.
target("vri-shaderc")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_rules("slangsdk")

    add_files("main.cpp")
target_end()
