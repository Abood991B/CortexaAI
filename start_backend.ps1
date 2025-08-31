# Start the FastAPI backend server
Set-Location $PSScriptRoot
$env:PYTHONPATH = "."
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
