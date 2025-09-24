@echo off
echo === Install Java JDK 17 (wajib untuk LanguageTool) ===
winget install Microsoft.OpenJDK.17
java -version

echo === Install FFmpeg (opsional) ===
winget install Gyan.FFmpeg
ffmpeg -version

echo === Buat virtual environment ===
python -m venv venv
call venv\Scripts\activate

echo === Install library Python ===
pip install --upgrade pip
pip install -r requirements.txt
