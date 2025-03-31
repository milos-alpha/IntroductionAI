@echo off
echo Traffic Sign Recognition System
echo =================================

:menu
echo.
echo Choose an option:
echo 1. Train the model
echo 2. Collect test images
echo 3. Run the prediction GUI
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto collect
if "%choice%"=="3" goto predict
if "%choice%"=="4" goto end

echo Invalid choice. Please try again.
goto menu

:train
echo.
echo Training the model...
python milton\traffic.py data\gtsrb best_model.h5
echo.
echo Training complete. Press any key to return to menu...
pause > nul
goto menu

:collect
echo.
echo Collecting test images...
python milton\collect_test_images.py
echo.
echo Collection complete. Press any key to return to menu...
pause > nul
goto menu

:predict
echo.
echo Running prediction GUI...
python milton\prediction_sign.py
goto menu

:end
echo.
echo Thank you for using the Traffic Sign Recognition System.
echo Exiting...
timeout 2 > nul