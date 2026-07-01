# Load .env
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        Write-Host "Loaded: $key" -ForegroundColor DarkGray
    }
}

$env:PYTHONUTF8 = "1"

uvicorn api.server:app --port 8000 --reload