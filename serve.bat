@echo off
REM HALO-GI standalone demo. WebGPU needs a secure context, so serve over
REM http://localhost (file:// will not work). Then open http://127.0.0.1:8777/
cd /d "%~dp0"
echo Serving HALO-GI demo at http://127.0.0.1:8777/  (Ctrl+C to stop)
python -m http.server 8777 --bind 127.0.0.1
