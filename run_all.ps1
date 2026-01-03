Set-Location -LiteralPath 'C:\Users\ayoub\Desktop\can2025-project'

# Stop any running Streamlit processes
Stop-Process -Name streamlit -ErrorAction SilentlyContinue -Force

# Activate venv
if (Test-Path -Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}

# Install requirements (idempotent)
python -m pip install -r requirements.txt

# Run pipeline steps
py -3 src\ingest.py
py -3 src\etl.py
py -3 src\features.py
py -3 src\model.py
py -3 src\evaluate.py

# List teams for debugging
py -3 scripts\list_teams.py

# Start Streamlit in background and return its PID
$p = Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList '-m','streamlit','run','src\dashboard.py' -NoNewWindow -PassThru
Write-Output "STREAMLIT_PID:$($p.Id)"
