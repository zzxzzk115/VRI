-- example-cuda-interop: VRI <-> CUDA external-memory interop demo.
--
-- Opt-in (off by default) because it needs the CUDA Toolkit (nvcc + cudart). The whole
-- target is guarded on the option so a normal build never tries to resolve CUDA.
--
-- CUDA comes from xmake-repo's `cuda` package: it first looks for a system install
-- (find_cuda / CUDA_PATH), and on Windows x64 can download+install the Toolkit if absent.
-- The package is a toolchain, so it provides nvcc plus the cudart include/link dirs.

if has_config("vri_build_cuda_interop") then

    add_requires("cuda", {optional = true, configs = {utils = {"cudart"}}})

    target("example-cuda-interop")
        set_kind("binary")
        set_languages("cxx23")
        set_default(false)

        add_deps("vri")

        add_files("main.cpp")      -- VRI host side (normal C++ toolchain)
        add_files("cuda_interop.cu") -- CUDA consumer side (compiled by nvcc)

        -- Use the required `cuda` package as the CUDA toolchain (provides nvcc + the SDK
        -- dirs). This works whether the package found a system install or installed its own,
        -- so it does not depend on a global CUDA_PATH being set.
        set_toolchains("@cuda")
        add_packages("cuda")       -- cudart include/link for the host TU
        add_cugencodes("native")   -- generate for the GPU in this machine
    target_end()

end
