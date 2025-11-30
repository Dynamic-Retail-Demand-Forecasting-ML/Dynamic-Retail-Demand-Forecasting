@echo off
echo ========================================
echo  Frontend Cleanup Script
echo ========================================
echo.
echo This will remove unnecessary files and folders from the Frontend directory.
echo.
echo Files/Folders to be removed:
echo   - Code folder (development files)
echo   - ML FINAL folder (training scripts)
echo   - Temporary command files ([32, cd, ls, pip, python)
echo.
echo Files/Folders to be KEPT:
echo   - app.py (Backend server)
echo   - requirements_backend.txt (Dependencies)
echo   - start_backend.bat (Startup script)
echo   - saved_models folder (ML models)
echo   - Presentation folder (Frontend UI)
echo   - .git folder (Version control)
echo.
pause
echo.
echo Starting cleanup...
echo.

REM Remove development folders
if exist "Code" (
    echo Removing Code folder...
    rmdir /s /q "Code"
    echo   ✓ Code folder removed
) else (
    echo   - Code folder not found
)

if exist "ML FINAL" (
    echo Removing ML FINAL folder...
    rmdir /s /q "ML FINAL"
    echo   ✓ ML FINAL folder removed
) else (
    echo   - ML FINAL folder not found
)

REM Remove temporary command files
if exist "[32" (
    echo Removing temporary file: [32
    del /q "[32"
    echo   ✓ [32 removed
)

if exist "cd" (
    echo Removing temporary file: cd
    del /q "cd"
    echo   ✓ cd removed
)

if exist "ls" (
    echo Removing temporary file: ls
    del /q "ls"
    echo   ✓ ls removed
)

if exist "pip" (
    echo Removing temporary file: pip
    del /q "pip"
    echo   ✓ pip removed
)

if exist "python" (
    echo Removing temporary file: python
    del /q "python"
    echo   ✓ python removed
)

echo.
echo ========================================
echo  Cleanup Complete!
echo ========================================
echo.
echo Your Frontend folder now contains only:
echo   - app.py (Backend server)
echo   - requirements_backend.txt (Dependencies)
echo   - start_backend.bat (Startup script)
echo   - saved_models/ (ML models - 8 .pkl files)
echo   - Presentation/ (Frontend UI)
echo   - .git/ (Version control)
echo.
echo The project is now clean and ready for deployment!
echo ========================================
echo.
pause
