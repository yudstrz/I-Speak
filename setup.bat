@echo off
echo ===========================
echo Installing Java JDK 17
echo ===========================
winget install Microsoft.OpenJDK.17 -h
java -version
echo.

echo ===========================
echo Installing FFmpeg
echo ===========================
winget install Gyan.FFmpeg -h
ffmpeg -version
echo.

echo ===========================
echo Installing Python Libraries
echo ===========================
pip install --upgrade pip
pip install -r requirements.txt --no-warn-script-location
echo.

echo ===========================
echo Downloading SpaCy model
echo ===========================
python -m spacy download en_core_web_sm
echo.

echo ===========================
echo Setup Completed!
echo ===========================
pause
