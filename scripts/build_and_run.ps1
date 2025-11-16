Param(
    [switch]$UseGPU = $true,
    [switch]$Rebuild = $false
)

# Quick helper to build and run the app on Windows PowerShell (WSL-enabled Docker Desktop recommended)
# Usage:
#  .\scripts\build_and_run.ps1 -UseGPU -Rebuild   # build GPU image and run
#  .\scripts\build_and_run.ps1 -UseGPU            # run existing image
#  .\scripts\build_and_run.ps1                    # run CPU image by default

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Project root is the parent directory of the scripts folder
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

# Ensure chroma_db named volume exists
if (-not (docker volume ls --format '{{.Name}}' | Select-String -Pattern '^chroma_db$' -Quiet)) {
    Write-Host "Creating named volume 'chroma_db'..."
    docker volume create chroma_db | Out-Null
}

if ($UseGPU) {
    $imageTag = 'aurora-qa-api:gpu'
    if ($Rebuild) {
        Write-Host "Building GPU image (this may take a while)..."
        docker build --progress=plain -f Dockerfile.gpu -t $imageTag .
    }
    Write-Host "Running GPU container (compose)..."
    docker compose -f docker-compose.gpu.yml up --detach --build
} else {
    $imageTag = 'aurora-qa-api:cpu'
    if ($Rebuild) {
        Write-Host "Building CPU image..."
        docker build -f Dockerfile -t $imageTag .
    }
    Write-Host "Running CPU container (docker run)..."
    docker run -d --env-file .env -p 8000:8000 -v chroma_db:/app/chroma_db -v ${PWD}\data:/app/data $imageTag
}

Write-Host "To tail logs: docker logs --tail 200 -f <container-id-or-name>"
