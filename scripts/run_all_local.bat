@echo off
REM NASA Exoplanet ML Models - Local Batch Execution Script (Windows)
REM Runs all models sequentially with unified pipeline

setlocal enabledelayedexpansion

echo ==================================================
echo NASA Exoplanet ML - Batch Training
echo Started: %date% %time%
echo ==================================================
echo.

REM Create necessary directories
echo Creating directories...
if not exist "artifacts" mkdir artifacts
if not exist "results" mkdir results
if not exist "artifacts\xgboost" mkdir artifacts\xgboost
if not exist "artifacts\random_forest" mkdir artifacts\random_forest
if not exist "artifacts\mlp" mkdir artifacts\mlp
if not exist "artifacts\logistic_regression" mkdir artifacts\logistic_regression
if not exist "artifacts\svm" mkdir artifacts\svm

REM Initialize counters
set SUCCESSFUL=0
set FAILED=0
set FAILED_MODELS=

REM Train XGBoost
echo.
echo ==================================================
echo Training: XGBoost
echo ==================================================
python -m src.models.xgboost.train --config configs/local.yaml
if !errorlevel! equ 0 (
    echo [SUCCESS] XGBoost completed successfully
    set /a SUCCESSFUL+=1
) else (
    echo [FAILED] XGBoost failed
    set /a FAILED+=1
    set FAILED_MODELS=!FAILED_MODELS! xgboost
)

REM Train Random Forest
echo.
echo ==================================================
echo Training: Random Forest
echo ==================================================
python -m src.models.random_forest.train --config configs/local.yaml
if !errorlevel! equ 0 (
    echo [SUCCESS] Random Forest completed successfully
    set /a SUCCESSFUL+=1
) else (
    echo [FAILED] Random Forest failed
    set /a FAILED+=1
    set FAILED_MODELS=!FAILED_MODELS! random_forest
)

REM Train MLP
echo.
echo ==================================================
echo Training: MLP
echo ==================================================
python -m src.models.mlp.train --config configs/local.yaml
if !errorlevel! equ 0 (
    echo [SUCCESS] MLP completed successfully
    set /a SUCCESSFUL+=1
) else (
    echo [FAILED] MLP failed
    set /a FAILED+=1
    set FAILED_MODELS=!FAILED_MODELS! mlp
)

REM Train Logistic Regression
echo.
echo ==================================================
echo Training: Logistic Regression
echo ==================================================
python -m src.models.logistic_regression.train --config configs/local.yaml
if !errorlevel! equ 0 (
    echo [SUCCESS] Logistic Regression completed successfully
    set /a SUCCESSFUL+=1
) else (
    echo [FAILED] Logistic Regression failed
    set /a FAILED+=1
    set FAILED_MODELS=!FAILED_MODELS! logistic_regression
)

REM Train SVM
echo.
echo ==================================================
echo Training: SVM
echo ==================================================
python -m src.models.svm.train --config configs/local.yaml
if !errorlevel! equ 0 (
    echo [SUCCESS] SVM completed successfully
    set /a SUCCESSFUL+=1
) else (
    echo [FAILED] SVM failed
    set /a FAILED+=1
    set FAILED_MODELS=!FAILED_MODELS! svm
)

REM Generate benchmark summary
echo.
echo ==================================================
echo Generating benchmark summary...
echo ==================================================

python scripts\generate_summary.py

if !errorlevel! equ 0 (
    echo [SUCCESS] Benchmark summary generated: results\benchmark_summary.md
) else (
    echo [WARNING] Failed to generate benchmark summary
)

REM Final summary
echo.
echo ==================================================
echo Batch Training Complete
echo ==================================================
echo Finished: %date% %time%
echo.
echo Successful: !SUCCESSFUL!
if !FAILED! gtr 0 (
    echo Failed: !FAILED!
    echo Failed models:!FAILED_MODELS!
)
echo.
echo Results saved to: results\benchmark_summary.md
echo ==================================================

REM Exit with error if any model failed
if !FAILED! gtr 0 (
    exit /b 1
)

exit /b 0
