$ErrorActionPreference = 'Stop'

# nvcc delegates host C++ compilation to MSVC on Windows. A normal PowerShell
# session may not have cl.exe on PATH, so initialize the latest VS environment.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) {
    throw 'Visual Studio Build Tools were not found. Install Desktop development with C++.'
  }
  $vsInstall = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
  $devShell = Join-Path $vsInstall 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
  $env:PATH = "$(Split-Path $vswhere);$env:PATH"
  Import-Module $devShell
  Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation `
    -DevCmdArguments '-arch=x64 -host_arch=x64'
}

$examples = @(
  '00_device_info',
  '01_hello_hierarchy',
  '02_vector_add_explicit',
  '03_vector_add_managed',
  '04_warp_divergence',
  '05_shared_reverse',
  '06_block_reduction',
  '07_stencil_2d',
  '09_cluster_support',
  '10_mapped_host_memory',
  '11_add_one_walkthrough'
)

foreach ($name in $examples) {
  Write-Host "`n== $name ==" -ForegroundColor Cyan
  $exe = "$name.exe"
  nvcc -std=c++17 -arch=native -lineinfo "$name.cu" -o $exe
  if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $name" }
  & ".\$exe"
  if ($LASTEXITCODE -ne 0) { throw "$name failed" }
}

Write-Host "`n== 08_tile_vector_add ==" -ForegroundColor Cyan
nvcc -std=c++20 -enable-tile -arch=native -lineinfo 08_tile_vector_add.cu -o 08_tile_vector_add.exe
if ($LASTEXITCODE -ne 0) { throw 'nvcc failed for tile example' }
& .\08_tile_vector_add.exe
