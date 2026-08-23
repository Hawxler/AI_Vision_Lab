$ErrorActionPreference = 'Stop'
$modelDir = Join-Path $PSScriptRoot 'models'
$modelPath = Join-Path $modelDir 'face_landmarker.task'
$modelUrl = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
Write-Host "Downloaded: $modelPath"
