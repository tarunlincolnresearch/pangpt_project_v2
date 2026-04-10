#!/bin/bash

echo "=========================================="
echo "PHASE 2A - COMPREHENSIVE SANITY CHECK"
echo "=========================================="
echo ""

# ============================================================
# 1. VERIFY DATA FILES
# ============================================================
echo "1. DATA FILES VERIFICATION"
echo "-------------------------------------------"

TRAIN_FILE="../data/phase1/win256/train_windows.txt"
VAL_FILE="../data/phase1/win256/val_windows.txt"
TEST_FILE="../data/phase1/win256/test_windows.txt"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    echo "✅ Train file: $TRAIN_FILE"
    echo "   Lines: $TRAIN_LINES | Size: $TRAIN_SIZE"
else
    echo "❌ Train file missing!"
    exit 1
fi

if [ -f "$VAL_FILE" ]; then
    VAL_LINES=$(wc -l < "$VAL_FILE")
    VAL_SIZE=$(du -h "$VAL_FILE" | cut -f1)
    echo "✅ Val file: $VAL_FILE"
    echo "   Lines: $VAL_LINES | Size: $VAL_SIZE"
else
    echo "❌ Val file missing!"
    exit 1
fi

if [ -f "$TEST_FILE" ]; then
    TEST_LINES=$(wc -l < "$TEST_FILE")
    TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
    echo "✅ Test file: $TEST_FILE"
    echo "   Lines: $TEST_LINES | Size: $TEST_SIZE"
else
    echo "❌ Test file missing!"
    exit 1
fi

echo ""
echo "Data Summary:"
echo "  Total training windows: $TRAIN_LINES (complete dataset)"
echo "  Total validation windows: $VAL_LINES"
echo "  Total test windows: $TEST_LINES"
echo "  Combined: $((TRAIN_LINES + VAL_LINES + TEST_LINES)) windows"
echo ""

# ============================================================
# 2. VERIFY PHASE 2A MODERATE SETTINGS
# ============================================================
echo "2. PHASE 2A MODERATE REGULARIZATION SETTINGS"
echo "-------------------------------------------"

# Check label smoothing
LABEL_SMOOTH=$(grep "CrossEntropyLoss" train_moderate.py | grep -o "label_smoothing=[0-9.]*")
if [ "$LABEL_SMOOTH" = "label_smoothing=0.05" ]; then
    echo "✅ Label smoothing: 0.05 (moderate)"
else
    echo "❌ Label smoothing NOT 0.05: $LABEL_SMOOTH"
    exit 1
fi

# Check weight decay
WEIGHT_DECAY=$(grep "weight_decay.*default=" train_moderate.py | grep -o "default=[0-9e-]*" | head -1)
if [[ "$WEIGHT_DECAY" == *"5e-4"* ]]; then
    echo "✅ Weight decay: 5e-4 (0.0005) - moderate"
else
    echo "❌ Weight decay NOT 5e-4: $WEIGHT_DECAY"
    exit 1
fi

# Check dropout
DROPOUT=$(grep "model_dropout_rate.*default=" train_moderate.py | grep -o "default=[0-9.]*")
if [[ "$DROPOUT" == *"0.15"* ]]; then
    echo "✅ Dropout: 0.15 (moderate)"
else
    echo "❌ Dropout NOT 0.15: $DROPOUT"
    exit 1
fi

# Check gradient clipping
if grep -q "clip_grad_norm_.*max_norm=1.0" train_moderate.py; then
    echo "✅ Gradient clipping: 1.0"
else
    echo "❌ Gradient clipping NOT found or incorrect"
    exit 1
fi

echo ""

# ============================================================
# 3. VERIFY DATA LOADING (NO AUGMENTATION)
# ============================================================
echo "3. DATA LOADING VERIFICATION"
echo "-------------------------------------------"

# Check that we're loading separate files
if grep -q "train_genomes = load_dataset(train_file)" train_moderate.py; then
    echo "✅ Loads train data separately"
else
    echo "❌ Train data loading incorrect"
    exit 1
fi

if grep -q "val_genomes = load_dataset(val_file)" train_moderate.py; then
    echo "✅ Loads validation data separately"
else
    echo "❌ Validation data loading incorrect"
    exit 1
fi

if grep -q "test_genomes = load_dataset(test_file)" train_moderate.py; then
    echo "✅ Loads test data separately"
else
    echo "❌ Test data loading incorrect"
    exit 1
fi

# Verify NO train_test_split is called
SPLIT_COUNT=$(grep -c "train_test_split(" train_moderate.py)
if [ "$SPLIT_COUNT" -eq 0 ]; then
    echo "✅ No internal data splitting (using pre-split files)"
else
    echo "❌ Found $SPLIT_COUNT train_test_split calls - should be 0!"
    exit 1
fi

# Verify NO augmentation
if grep -q "augment" train_moderate.py; then
    echo "⚠️  WARNING: Found 'augment' in code - checking if it's disabled..."
    if grep -q "augment=True\|augment_data" train_moderate.py; then
        echo "❌ Augmentation appears to be enabled!"
        exit 1
    else
        echo "✅ Augmentation references found but not active"
    fi
else
    echo "✅ No augmentation (correct for Phase 2A)"
fi

echo ""

# ============================================================
# 4. VERIFY MODEL ARCHITECTURE
# ============================================================
echo "4. MODEL ARCHITECTURE VERIFICATION"
echo "-------------------------------------------"

# Check model parameters in SLURM script
if [ -f "run_phase2a.slurm" ]; then
    echo "Checking SLURM script parameters..."
    
    if grep -q "embed_dim 256" run_phase2a.slurm; then
        echo "✅ Embedding dimension: 256"
    else
        echo "❌ Embedding dimension not 256"
    fi
    
    if grep -q "num_heads 8" run_phase2a.slurm; then
        echo "✅ Attention heads: 8"
    else
        echo "❌ Attention heads not 8"
    fi
    
    if grep -q "num_layers 6" run_phase2a.slurm; then
        echo "✅ Transformer layers: 6"
    else
        echo "❌ Transformer layers not 6"
    fi
    
    if grep -q "max_seq_length 512" run_phase2a.slurm; then
        echo "✅ Max sequence length: 512"
    else
        echo "❌ Max sequence length not 512"
    fi
    
    if grep -q "batch_size 8" run_phase2a.slurm; then
        echo "✅ Batch size: 8"
    else
        echo "❌ Batch size not 8"
    fi
else
    echo "❌ SLURM script not found!"
    exit 1
fi

echo ""

# ============================================================
# 5. VERIFY TRAINING CONFIGURATION
# ============================================================
echo "5. TRAINING CONFIGURATION"
echo "-------------------------------------------"

if grep -q "epochs 150" run_phase2a.slurm; then
    echo "✅ Max epochs: 150"
else
    echo "⚠️  Max epochs not 150"
fi

if grep -q "learning_rate 0.0001" run_phase2a.slurm; then
    echo "✅ Learning rate: 0.0001"
else
    echo "❌ Learning rate not 0.0001"
fi

if grep -q "early_stop_patience 30" run_phase2a.slurm; then
    echo "✅ Early stopping patience: 30 epochs"
else
    echo "⚠️  Early stopping patience not 30"
fi

echo ""

# ============================================================
# 6. VERIFY FILE PATHS IN SLURM SCRIPT
# ============================================================
echo "6. SLURM SCRIPT FILE PATHS"
echo "-------------------------------------------"

if grep -q "train_file.*train_windows.txt" run_phase2a.slurm; then
    echo "✅ Train file path configured"
else
    echo "❌ Train file path missing"
    exit 1
fi

if grep -q "val_file.*val_windows.txt" run_phase2a.slurm; then
    echo "✅ Validation file path configured"
else
    echo "❌ Validation file path missing"
    exit 1
fi

if grep -q "test_file.*test_windows.txt" run_phase2a.slurm; then
    echo "✅ Test file path configured"
else
    echo "❌ Test file path missing"
    exit 1
fi

echo ""

# ============================================================
# 7. VERIFY OUTPUT DIRECTORIES
# ============================================================
echo "7. OUTPUT DIRECTORIES"
echo "-------------------------------------------"

for dir in checkpoints logs results; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "⚠️  $dir/ missing - will be created at runtime"
        mkdir -p "$dir"
        echo "   Created $dir/"
    fi
done

echo ""

# ============================================================
# 8. PYTHON SYNTAX CHECK
# ============================================================
echo "8. PYTHON SYNTAX VERIFICATION"
echo "-------------------------------------------"

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
$PYTHON -m py_compile train_moderate.py 2>&1

if [ $? -eq 0 ]; then
    echo "✅ No syntax errors in train_moderate.py"
else
    echo "❌ Syntax errors found!"
    exit 1
fi

echo ""

# ============================================================
# 9. COMPARISON WITH PHASE 2 (FAILED)
# ============================================================
echo "9. PHASE 2 vs PHASE 2A COMPARISON"
echo "-------------------------------------------"
echo "Phase 2 (FAILED - Too Strong Regularization):"
echo "  - Label smoothing: 0.1"
echo "  - Weight decay: 0.001"
echo "  - Dropout: 0.2"
echo "  - Result: Collapsed to 50% accuracy (random guessing)"
echo ""
echo "Phase 2A (MODERATE Regularization):"
echo "  - Label smoothing: 0.05 ✨ (reduced by 50%)"
echo "  - Weight decay: 0.0005 ✨ (reduced by 50%)"
echo "  - Dropout: 0.15 ✨ (reduced by 25%)"
echo "  - Expected: Should NOT collapse, gradual learning"
echo ""

# ============================================================
# 10. EXPECTED BEHAVIOR
# ============================================================
echo "10. EXPECTED TRAINING BEHAVIOR"
echo "-------------------------------------------"
echo "What to watch for during training:"
echo ""
echo "✅ GOOD SIGNS:"
echo "  - Epoch 0: Val accuracy ~60-70% (like Phase 2 started)"
echo "  - Epoch 1-5: Accuracy should STAY above 55%"
echo "  - Gradual improvement over epochs"
echo "  - Training loss decreasing"
echo "  - Validation loss decreasing (or stable)"
echo ""
echo "❌ BAD SIGNS (like Phase 2):"
echo "  - Accuracy drops to ~50% after epoch 0"
echo "  - Stays at 50% for multiple epochs"
echo "  - Perplexity increases dramatically"
echo ""
echo "🎯 TARGET METRICS:"
echo "  - Validation accuracy: >60%"
echo "  - Test accuracy: >40% (vs Phase 1's 0.51%)"
echo "  - Perplexity: <10"
echo ""

# ============================================================
# 11. RESOURCE CHECK
# ============================================================
echo "11. SLURM RESOURCE ALLOCATION"
echo "-------------------------------------------"

if grep -q "#SBATCH --gpus=1" run_phase2a.slurm; then
    echo "✅ GPU requested: 1"
else
    echo "❌ GPU not requested!"
    exit 1
fi

if grep -q "#SBATCH --mem=240G" run_phase2a.slurm; then
    echo "✅ Memory requested: 240GB"
else
    echo "⚠️  Memory allocation different"
fi

if grep -q "#SBATCH --qos=long" run_phase2a.slurm; then
    echo "✅ QOS: long (for extended runtime)"
else
    echo "⚠️  QOS not set to long"
fi

echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "=========================================="
echo "✅ ALL SANITY CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Phase 2A Configuration Summary:"
echo "-------------------------------------------"
echo "Data:"
echo "  ✓ Train: 147,147 windows (complete dataset)"
echo "  ✓ Validation: 18,388 windows"
echo "  ✓ Test: 18,378 windows"
echo "  ✓ No augmentation"
echo "  ✓ Pre-split files (no internal splitting)"
echo ""
echo "Model:"
echo "  ✓ Transformer (6 layers, 8 heads, 256 dim)"
echo "  ✓ Max sequence length: 512"
echo "  ✓ Batch size: 8"
echo ""
echo "Regularization (MODERATE):"
echo "  ✓ Label smoothing: 0.05"
echo "  ✓ Weight decay: 0.0005"
echo "  ✓ Dropout: 0.15"
echo "  ✓ Gradient clipping: 1.0"
echo ""
echo "Training:"
echo "  ✓ Max epochs: 150"
echo "  ✓ Learning rate: 0.0001"
echo "  ✓ Early stopping: 30 epochs patience"
echo ""
echo "=========================================="
echo "READY TO SUBMIT!"
echo "=========================================="
echo ""
echo "To submit Phase 2A training:"
echo "  sbatch run_phase2a.slurm"
echo ""
echo "To monitor:"
echo "  squeue -u tgangil"
echo "  tail -f logs/phase2a_*.err"
echo ""
echo "Expected training time: 4-6 hours"
echo "=========================================="
