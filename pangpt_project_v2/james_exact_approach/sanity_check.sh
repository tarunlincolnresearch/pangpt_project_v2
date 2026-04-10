#!/bin/bash

# ============================================================
# COMPLETE SANITY CHECK - panGPT Training Environment
# ============================================================

echo "============================================================"
echo "SANITY CHECK STARTED: $(date)"
echo "============================================================"
echo ""

ERRORS=0
WARNINGS=0

# ── 1. Check Current Directory ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CHECKING CURRENT DIRECTORY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"
if [[ "$CURRENT_DIR" == *"james_exact_approach"* ]]; then
    echo "✅ Correct directory"
else
    echo "❌ ERROR: Not in james_exact_approach directory!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ── 2. Check Required Directories ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. CHECKING REQUIRED DIRECTORIES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for dir in checkpoints data logs; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "❌ ERROR: $dir/ directory missing!"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ── 3. Check Input Data ───────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. CHECKING INPUT DATA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
INPUT_FILE="data/all_genomes.txt"
if [ -f "$INPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
    LINE_COUNT=$(wc -l < "$INPUT_FILE")
    echo "✅ $INPUT_FILE exists"
    echo "   Size: $FILE_SIZE"
    echo "   Lines: $LINE_COUNT"
    
    if [ "$LINE_COUNT" -lt 10 ]; then
        echo "⚠️  WARNING: Very few lines in input file!"
        WARNINGS=$((WARNINGS + 1))
    fi
    
    # Show first 3 lines
    echo "   First 3 lines:"
    head -n 3 "$INPUT_FILE" | sed 's/^/   | /'
else
    echo "❌ ERROR: $INPUT_FILE not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ── 4. Check Python Scripts ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. CHECKING PYTHON SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for script in panGPT.py panPrompt.py movingSplits.py prepare_input_for_james.py; do
    if [ -f "$script" ]; then
        echo "✅ $script exists"
    else
        echo "❌ ERROR: $script missing!"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ── 5. Check SLURM Scripts ────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. CHECKING SLURM SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for script in run_james_original.slurm run_james_improved.slurm run_james_quick_test.slurm; do
    if [ -f "$script" ]; then
        echo "✅ $script exists"
        # Check if it has execute permissions
        if [ -x "$script" ]; then
            echo "   (executable)"
        fi
    else
        echo "❌ ERROR: $script missing!"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ── 6. Check Utility Scripts ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. CHECKING UTILITY SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for script in monitor_job.sh; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✅ $script exists and is executable"
        else
            echo "⚠️  WARNING: $script exists but not executable"
            echo "   Run: chmod +x $script"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "❌ ERROR: $script missing!"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ── 7. Check Python Environment ───────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. CHECKING PYTHON ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python

if [ -f "$PYTHON" ]; then
    echo "✅ Python binary exists: $PYTHON"
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON --version 2>&1)
    echo "   Version: $PYTHON_VERSION"
    
    # Check if Python works
    if $PYTHON -c "print('Python works')" &>/dev/null; then
        echo "✅ Python executable works"
    else
        echo "❌ ERROR: Python executable doesn't work!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ ERROR: Python binary not found at $PYTHON"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ── 8. Check Python Packages ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. CHECKING PYTHON PACKAGES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$PYTHON" ]; then
    for package in torch transformers numpy pandas; do
        if $PYTHON -c "import $package; print('$package version:', $package.__version__)" 2>/dev/null; then
            echo "✅ $package installed"
        else
            echo "❌ ERROR: $package not installed!"
            ERRORS=$((ERRORS + 1))
        fi
    done
else
    echo "⏭️  Skipping (Python not available)"
fi
echo ""

# ── 9. Check PyTorch CUDA Support ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "9. CHECKING PYTORCH CUDA SUPPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$PYTHON" ]; then
    CUDA_CHECK=$($PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
    
    if [[ "$CUDA_CHECK" == *"True"* ]]; then
        echo "✅ PyTorch CUDA support available"
        echo "$CUDA_CHECK" | sed 's/^/   /'
    else
        echo "⚠️  WARNING: PyTorch CUDA not available (will use CPU)"
        echo "$CUDA_CHECK" | sed 's/^/   /'
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "⏭️  Skipping (Python not available)"
fi
echo ""

# ── 10. Check Modules ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "10. CHECKING AVAILABLE MODULES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v module &> /dev/null; then
    echo "✅ Module system available"
    
    # Check for required modules
    for mod in gcc nvidia/cuda; do
        if module avail $mod 2>&1 | grep -q "$mod"; then
            echo "✅ Module available: $mod"
        else
            echo "⚠️  WARNING: Module not found: $mod"
            WARNINGS=$((WARNINGS + 1))
        fi
    done
else
    echo "⚠️  WARNING: Module system not available (might be on login node)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ── 11. Check GPU Partition ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "11. CHECKING GPU PARTITION AVAILABILITY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v sinfo &> /dev/null; then
    GPU_INFO=$(sinfo -p gpu -o "%P %a %l %D %N" 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        echo "✅ GPU partition available"
        echo "$GPU_INFO" | sed 's/^/   /'
    else
        echo "⚠️  WARNING: GPU partition info not available"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "⚠️  WARNING: sinfo command not available"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ── 12. Check Disk Space ──────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "12. CHECKING DISK SPACE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DISK_USAGE=$(df -h. | tail -1)
DISK_PERCENT=$(echo "$DISK_USAGE" | awk '{print $5}' | sed 's/%//')
echo "Current directory disk usage:"
echo "$DISK_USAGE" | sed 's/^/   /'

if [ "$DISK_PERCENT" -gt 90 ]; then
    echo "❌ ERROR: Disk usage above 90%!"
    ERRORS=$((ERRORS + 1))
elif [ "$DISK_PERCENT" -gt 80 ]; then
    echo "⚠️  WARNING: Disk usage above 80%"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✅ Sufficient disk space"
fi
echo ""

# ── 13. Check Quota (if available) ────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "13. CHECKING DISK QUOTA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v lsquota &> /dev/null; then
    echo "Quota information:"
    lsquota | sed 's/^/   /'
else
    echo "⏭️  lsquota command not available"
fi
echo ""

# ── 14. Verify SLURM Script Syntax ────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "14. VERIFYING SLURM SCRIPT SYNTAX"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for script in run_james_improved.slurm run_james_quick_test.slurm; do
    if [ -f "$script" ]; then
        # Check for required SBATCH directives
        if grep -q "#SBATCH --job-name" "$script" && \
           grep -q "#SBATCH --output" "$script" && \
           grep -q "#SBATCH --partition=gpu" "$script"; then
            echo "✅ $script has required SBATCH directives"
        else
            echo "⚠️  WARNING: $script might be missing SBATCH directives"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
done
echo ""

# ── 15. Check for Running Jobs ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "15. CHECKING FOR RUNNING JOBS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v squeue &> /dev/null; then
    RUNNING_JOBS=$(squeue -u tgangil 2>/dev/null | tail -n +2 | wc -l)
    if [ "$RUNNING_JOBS" -gt 0 ]; then
        echo "⚠️  WARNING: You have $RUNNING_JOBS running/pending job(s)"
        squeue -u tgangil | sed 's/^/   /'
        WARNINGS=$((WARNINGS + 1))
    else
        echo "✅ No running jobs"
    fi
else
    echo "⏭️  squeue command not available"
fi
echo ""

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "SANITY CHECK SUMMARY"
echo "============================================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED!"
    echo ""
    echo "✅ Everything looks good. You're ready to start training!"
    echo ""
    echo "Recommended next steps:"
    echo "  1. Run quick test:  sbatch run_james_quick_test.slurm"
    echo "  2. Monitor:./monitor_job.sh"
    echo "  3. View output:     tail -f logs/james_test_*.out"
    echo ""
    EXIT_CODE=0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  CHECKS PASSED WITH WARNINGS"
    echo ""
    echo "Warnings: $WARNINGS"
    echo ""
    echo "You can proceed, but review the warnings above."
    echo ""
    EXIT_CODE=0
else
    echo "❌ CHECKS FAILED"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo ""
    EXIT_CODE=1
fi

echo "============================================================"
echo "SANITY CHECK COMPLETED: $(date)"
echo "============================================================"

exit $EXIT_CODE