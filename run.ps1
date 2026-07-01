# Load .env file and inject into current shell session
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        Write-Host "Loaded: $key" -ForegroundColor DarkGray
    }
}

# Set UTF8 to avoid emoji errors
$env:PYTHONUTF8 = "1"

# Run FENRIR with any args you pass
python -m fenrir @args