# CUDA foundations lab files

Open PowerShell in this directory, then run:

```powershell
nvcc --version
nvidia-smi
Set-ExecutionPolicy -Scope Process Bypass
.\build_and_run.ps1
```

On Windows, `nvcc` uses Microsoft's `cl.exe` for host code. The lab runner
automatically initializes the latest installed Visual Studio C++ environment.
If compiling manually reports `Cannot find compiler 'cl.exe'`, either use the
runner or open **x64 Native Tools Command Prompt for VS** first.

For one lab at a time:

```powershell
nvcc -std=c++17 -arch=native -lineinfo 02_vector_add_explicit.cu -o vector_add.exe
.\vector_add.exe
compute-sanitizer --tool memcheck .\vector_add.exe
```

CUDA Tile C++ is new in Toolkit 13.3 and requires C++20:

```powershell
nvcc -std=c++20 -enable-tile -arch=native 08_tile_vector_add.cu -o tile_vector_add.exe
.\tile_vector_add.exe
```

`cuda_check.cuh` supplies the generic CUDA Runtime error-checking macro used by
most CUDA examples. The two CPU examples are optional Module 2 companions and
require an OpenMP-capable C++ compiler or an MPI implementation, respectively.

Start with `11_add_one_walkthrough.cu` if CUDA and C++ pointer-based memory APIs
are both new to you. It is deliberately self-contained and spells out its
headers and error-checking function instead of using `cuda_check.cuh`.

`10_mapped_host_memory.cu` demonstrates zero-copy access: CUDA allocates pinned
CPU memory, creates a device-visible mapping, and a kernel modifies it across
the system interconnect. This demonstrates accessibility, not a claim that
remote memory is as fast as the GPU's own DRAM.
