@echo off
rem Self-tuning training loop. Restarts the bot automatically if it crashes.
rem F12 in the game quits for good. Extra arguments are passed through (e.g. --debug).
cd /d "%~dp0"
:loop
python -m slither_bot.main --train --start %*
if %errorlevel%==0 goto :eof
echo [train.bat] bot crashed (exit %errorlevel%), restarting in 5 s ...
timeout /t 5 >nul
goto loop
