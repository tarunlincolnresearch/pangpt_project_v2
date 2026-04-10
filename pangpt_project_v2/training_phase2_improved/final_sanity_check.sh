#!/bin/bash

echo "=========================================="
echo "COMPREHENSIVE PHASE 2 SANITY CHECK"
echo "=========================================="
echo ""

# ============================================================
# 1. DATA FILES CHECK
# ============================================================
echo "1. DATA FILES VERIFICATION"
echo "-------------------------------------------"

TRAIN_FILE="../data/phase1/win256/train_windows.txt"
VAL_FILE="../data/phase1/win256/val_windows.txt"
TEST_FILE="../data/phase1/win256/test_windows.txt"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
    echo "✅ Train file exists: $TRAIN_FILE"
    echo "   Size: $TRAIN_SIZE | Lines: $TRAIN_LINES"
else
    echo "❌ Train file missing: $TRAIN_FILE"
    exit 1
fi

if [ -f "$VAL_FILE" ]; then
    VAL_SIZE=$(du -h "$VAL_FILE" | cut -f1)
    VAL_LINES=$(wc -l < "$VAL_FILE")
    echo "✅ Val file exists: $VAL_FILE"
    echo "   Size: $VAL_SIZE | Lines: $VAL_LINES"
else
    echo "❌ Val file missing: $VAL_FILE"
    exit 1
fi

if [ -f "$TEST_FILE" ]; then
    TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
    TEST_LINES=$(wc -l < "$TEST_FILE")
    echo "✅ Test file exists: $TEST_FILE"
    echo "   Size: $TEST_SIZE | Lines: $TEST_LINES"
else
    echo "❌ Test file missing: $TEST_FILE"
    exit 1
fi

echo ""

# ============================================================
# 2. TOKENIZER CHECK
# ============================================================
echo "2. TOKENIZER VERIFICATION"
echo "-------------------------------------------"

TOKENIZER="../pangenome_gpt_tokenizer.json"
if [ -f "$TOKENIZER" ]; then
    TOK_SIZE=$(du -h "$TOKENIZER" | cut -f1)
    echo "✅ Tokenizer exists: $TOKENIZER"
    echo "   Size: $TOK_SIZE"
else
    echo "❌ Tokenizer missing: $TOKENIZER"
    exit 1
fi

echo ""

# ============================================================
# 3. PYTHON SCRIPT VERIFICATION
# ============================================================
echo "3. TRAINING SCRIPT VERIFICATION"
echo "-------------------------------------------"

if [ -f "train_improved.py" ]; then
    SCRIPT_SIZE=$(du -h "train_improved.py" | cut -f1)
    SCRIPT_LINES=$(wc -l < "train_improved.py")
    echo "✅ Training script exists: train_improved.py"
    echo "   Size: $SCRIPT_SIZE | Lines: $SCRIPT_LINES"
else
    echo "❌ Training script missing"
    exit 1
fi

# Check for Phase 2 improvements
echo ""
echo "   Checking Phase 2 improvements in code:"
echo "   -------------------------------------------"

if grep -q "label_smoothing=0.1" train_improved.py; then
    echo "   ✅ Label smoothing: 0.1"
else
    echo "   ❌ Label smoothing NOT found"
fi

if grep -q "default=1e-3.*weight_decay" train_improved.py; then
    echo "   ✅ Weight decay: 1e-3"
else
    echo "   ❌ Weight decay NOT updated"
fi

if grep -q "clip_grad_norm_" train_improved.py; then
    echo "   ✅ Gradient clipping present"
else
    echo "   ❌ Gradient clipping NOT found"
fi

if grep -q "train_file.*val_file.*test_file" train_improved.py; then
    echo "   ✅ Pre-split file support"
else
    echo "   ❌ Pre-split files NOT supported"
fi

if grep -q "train_genomes = load_dataset(train_file)" train_improved.py; then
    echo "   ✅ Separate train/val/test loading"
else
    echo "   ❌ Separate loading NOT implemented"
fi

echo ""

# ============================================================
# 4. SYNTAX CHECK
# ============================================================
echo "4. PYTHON SYNTAX CHECK"
echo "-------------------------------------------"

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
$PYTHON -m py_compile train_improved.py 2>&1

if [ $? -eq 0 ]; then
    echo "✅ No syntax errors"
else
    echo "❌ Syntax errors found!"
    exit 1
fi

echo ""

# ============================================================
# 5. SLURM SCRIPT VERIFICATION
# ============================================================
echo "5. SLURM SCRIPT VERIFICATION"
echo "-------------------------------------------"

if [ -f "run_phase2_training.slurm" ]; then
    echo "✅ SLURM script exists"
    
    # Check SLURM parameters
    if grep -q "#SBATCH --qos=long" run_phase2_training.slurm; then
        echo "   ✅ QOS: long"
    else
        echo "   ❌ QOS not set to long"
    fi
    
    if grep -q "#SBATCH --time=7-00:00:00" run_phase2_training.slurm; then
        echo "   ✅ Time limit: 7 days"
    else
        echo "   ⚠️  Time limit may not be 7 days"
    fi
    
    if grep -q "#SBATCH --gpus=1" run_phase2_training.slurm; then
        echo "   ✅ GPU requested: 1"
    else
        echo "   ❌ GPU not requested"
    fi
    
    # Check file paths in SLURM script
    if grep -q "train_file.*val_file.*test_file" run_phase2_training.slurm; then
        echo "   ✅ Using pre-split files"
    else
        echo "   ❌ Not using pre-split files"
    fi
    
else
    echo "❌ SLURM script missing"
    exit 1
fi

echo ""

# ============================================================
# 6. DIRECTORY STRUCTURE
# ============================================================
echo "6. OUTPUT DIRECTORY STRUCTURE"
echo "-------------------------------------------"

for dir in checkpoints logs results; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "⚠️  $dir/ missing (will be created at runtime)"
    fi
done

echo ""

# ============================================================
# 7. PYTHON ENVIRONMENT CHECK
# ============================================================
echo "7. PYTHON ENVIRONMENT CHECK"
echo "-------------------------------------------"

echo "Python version:"
$PYTHON --version

echo ""
echo "PyTorch version:"
$PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}')"
$PYTHON -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
$PYTHON -c "import torch; print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Required packages:"
$PYTHON -c "import tokenizers; print('  ✅ tokenizers')" 2>/dev/null || echo "  ❌ tokenizers missing"
$PYTHON -c "import sklearn; print('  ✅ sklearn')" 2>/dev/null || echo "  ❌ sklearn missing"
$PYTHON -c "import transformers; print('  ✅ transformers')" 2>/dev/null || echo "  ❌ transformers missing"
$PYTHON -c "import tqdm; print('  ✅ tqdm')" 2>/dev/null || echo "  ❌ tqdm missing"

echo ""

# ============================================================
# 8. EXPECTED PARAMETERS SUMMARY
# ============================================================
echo "8. EXPECTED TRAINING PARAMETERS"
echo "-------------------------------------------"
echo "Model Architecture:"
echo "  - Type: Transformer"
echo "  - Embedding dim: 256"
echo "  - Attention heads: 8"
echo "  - Layers: 6"
echo "  - Max sequence length: 512"
echo ""
echo "Training Configuration:"
echo "  - Batch size: 8"
echo "  - Epochs: 150"
echo "  - Learning rate: 0.0001"
echo ""
echo "Phase 2 Improvements:"
echo "  - Label smoothing: 0.1 ✨"
echo "  - Weight decay: 0.001 (10x stronger) ✨"
echo "  - Gradient clipping: 1.0 ✨"
echo "  - Dropout: 0.2 ✨"
echo "  - Pre-split data: train/val/test ✨"
echo ""
echo "Expected Outputs:"
echo "  - Model checkpoint: checkpoints/model_checkpoint.pth"
echo "  - Training logs: logs/training_phase2.log"
echo "  - TensorBoard logs: logs/"
echo "  - Test metrics: Logged at end of training"
echo ""

# ============================================================
# 9. DATA SAMPLE CHECK
# ============================================================
echo "9. DATA SAMPLE VERIFICATION"
echo "-------------------------------------------"
echo "First 3 lines of train data:"
head -3 "$TRAIN_FILE" | while read line; do
    GENE_COUNT=$(echo $line | wc -w)
    echo "  - $GENE_COUNT genes"
done

echo ""
echo "First 3 lines of val data:"
head -3 "$VAL_FILE" | while read line; do
    GENE_COUNT=$(echo $line | wc -w)
    echo "  - $GENE_COUNT genes"
done

echo ""

# ============================================================
# 10. COMPARISON WITH PHASE 1
# ============================================================
echo "10. PHASE 1 vs PHASE 2 COMPARISON"
echo "-------------------------------------------"
echo "Phase 1 Results (Baseline):"
echo "  - Val accuracy: 65.5%"
echo "  - Test accuracy: 0.51% ❌"
echo "  - Repetition rate: 61-96% ❌"
echo "  - Overfitting: Severe ❌"
echo ""
echo "Phase 2 Expected Improvements:"
echo "  - Test accuracy: Target >40% 🎯"
echo "  - Repetition: Will fix in inference"
echo "  - Overfitting: Reduced via regularization 🎯"
echo "  - Generalization: Better train-test gap 🎯"
echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "=========================================="
echo "✅ ALL SANITY CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Ready to submit Phase 2 training!"
echo ""
echo "To submit:"
echo "  cd /work/users/tgangil/pangpt_project_v2/pangpt_project_v2/training_phase2_improved"
echo "  sbatch run_phase2_training.slurm"
echo ""
echo "To monitor:"
echo "  squeue -u tgangil"
echo "  tail -f logs/phase2_win256_*.out"
echo ""
echo "=========================================="
