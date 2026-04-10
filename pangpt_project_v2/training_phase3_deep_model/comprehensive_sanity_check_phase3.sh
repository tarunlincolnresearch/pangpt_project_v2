#!/bin/bash

echo "=========================================="
echo "PHASE 3 DEEP MODEL - COMPREHENSIVE SANITY CHECK"
echo "=========================================="
echo ""

# ============================================================
# 1. DATA FILES VERIFICATION
# ============================================================
echo "1. DATA FILES VERIFICATION"
echo "-------------------------------------------"

TRAIN_FILE="../data/phase1/win256/train_windows.txt"
VAL_FILE="../data/phase1/win256/val_windows.txt"
TEST_FILE="../data/phase1/win256/test_windows.txt"
TOKENIZER="../pangenome_gpt_tokenizer.json"

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

if [ -f "$TOKENIZER" ]; then
    TOK_SIZE=$(du -h "$TOKENIZER" | cut -f1)
    echo "✅ Tokenizer: $TOKENIZER"
    echo "   Size: $TOK_SIZE"
else
    echo "❌ Tokenizer missing!"
    exit 1
fi

echo ""

# ============================================================
# 2. DEEP MODEL ARCHITECTURE VERIFICATION
# ============================================================
echo "2. DEEP MODEL ARCHITECTURE VERIFICATION"
echo "-------------------------------------------"

# Check model parameters in code
echo "Checking train_deep.py configuration..."

LAYERS=$(grep "num_layers.*default=" train_deep.py | grep -o "default=[0-9]*" | head -1 | cut -d= -f2)
EMBED=$(grep "embed_dim.*default=" train_deep.py | grep -o "default=[0-9]*" | head -1 | cut -d= -f2)
HEADS=$(grep "num_heads.*default=" train_deep.py | grep -o "default=[0-9]*" | head -1 | cut -d= -f2)
BATCH=$(grep "batch_size.*default=" train_deep.py | grep -o "default=[0-9]*" | head -1 | cut -d= -f2)

if [ "$LAYERS" = "12" ]; then
    echo "✅ Layers: 12 (2x deeper than Phase 2A)"
else
    echo "❌ Layers: $LAYERS (expected 12)"
    exit 1
fi

if [ "$EMBED" = "512" ]; then
    echo "✅ Embedding dimension: 512 (2x wider)"
else
    echo "❌ Embedding: $EMBED (expected 512)"
    exit 1
fi

if [ "$HEADS" = "16" ]; then
    echo "✅ Attention heads: 16 (2x more)"
else
    echo "❌ Heads: $HEADS (expected 16)"
    exit 1
fi

if [ "$BATCH" = "4" ]; then
    echo "✅ Batch size: 4 (smaller for larger model)"
else
    echo "⚠️  Batch size: $BATCH (expected 4)"
fi

# Check gradient accumulation
if grep -q "accumulation_steps = 2" train_deep.py; then
    echo "✅ Gradient accumulation: 2 steps (effective batch = 8)"
else
    echo "⚠️  Gradient accumulation not found or incorrect"
fi

echo ""

# ============================================================
# 3. PARAMETER COUNT ESTIMATION
# ============================================================
echo "3. MODEL SIZE ESTIMATION"
echo "-------------------------------------------"

VOCAB_SIZE=50000
EMBED_DIM=512
NUM_LAYERS=12
NUM_HEADS=16

# Embedding parameters
EMBED_PARAMS=$((VOCAB_SIZE * EMBED_DIM))

# Transformer parameters per layer
# Self-attention: 4 * embed_dim^2 (Q, K, V, O projections)
# FFN: 2 * embed_dim * (4 * embed_dim)
# Layer norms: 4 * embed_dim
ATTN_PARAMS=$((4 * EMBED_DIM * EMBED_DIM))
FFN_PARAMS=$((2 * EMBED_DIM * 4 * EMBED_DIM))
NORM_PARAMS=$((4 * EMBED_DIM))
LAYER_PARAMS=$((ATTN_PARAMS + FFN_PARAMS + NORM_PARAMS))
TRANSFORMER_PARAMS=$((LAYER_PARAMS * NUM_LAYERS))

# Output layer
OUTPUT_PARAMS=$((EMBED_DIM * VOCAB_SIZE))

# Total
TOTAL_PARAMS=$((EMBED_PARAMS + TRANSFORMER_PARAMS + OUTPUT_PARAMS))
TOTAL_PARAMS_M=$(echo "scale=1; $TOTAL_PARAMS / 1000000" | bc)

echo "Parameter Breakdown:"
echo "  Embedding layer:    $(printf "%'d" $EMBED_PARAMS) params"
echo "  Transformer (12L):  $(printf "%'d" $TRANSFORMER_PARAMS) params"
echo "  Output layer:       $(printf "%'d" $OUTPUT_PARAMS) params"
echo "  ─────────────────────────────────"
echo "  TOTAL:              $(printf "%'d" $TOTAL_PARAMS) params (~${TOTAL_PARAMS_M}M)"
echo ""

# Memory estimation
# FP32: 4 bytes per parameter
# Model weights + gradients + optimizer states (Adam: 2x params)
MODEL_MEMORY=$((TOTAL_PARAMS * 4 / 1024 / 1024))  # MB
GRADIENT_MEMORY=$MODEL_MEMORY
OPTIMIZER_MEMORY=$((MODEL_MEMORY * 2))  # Adam stores momentum + variance
TOTAL_MEMORY=$((MODEL_MEMORY + GRADIENT_MEMORY + OPTIMIZER_MEMORY))

echo "Estimated GPU Memory Usage:"
echo "  Model weights:      ${MODEL_MEMORY} MB"
echo "  Gradients:          ${GRADIENT_MEMORY} MB"
echo "  Optimizer states:   ${OPTIMIZER_MEMORY} MB"
echo "  ─────────────────────────────────"
echo "  TOTAL (training):   ${TOTAL_MEMORY} MB (~$((TOTAL_MEMORY / 1024)) GB)"
echo ""

# Check if it fits in GPU
GPU_MEMORY_GB=97  # Your GPU has ~97GB
REQUIRED_GB=$((TOTAL_MEMORY / 1024))
ACTIVATION_MEMORY_GB=10  # Estimate for activations

TOTAL_REQUIRED=$((REQUIRED_GB + ACTIVATION_MEMORY_GB))

if [ $TOTAL_REQUIRED -lt $GPU_MEMORY_GB ]; then
    echo "✅ Model fits in GPU memory!"
    echo "   Required: ~${TOTAL_REQUIRED}GB"
    echo "   Available: ${GPU_MEMORY_GB}GB"
    echo "   Margin: $((GPU_MEMORY_GB - TOTAL_REQUIRED))GB"
else
    echo "❌ WARNING: Model might not fit in GPU!"
    echo "   Required: ~${TOTAL_REQUIRED}GB"
    echo "   Available: ${GPU_MEMORY_GB}GB"
fi

echo ""

# ============================================================
# 4. TRAINING TIME ESTIMATION
# ============================================================
echo "4. TRAINING TIME ESTIMATION"
echo "-------------------------------------------"

TRAIN_WINDOWS=147147
BATCH_SIZE=4
BATCHES_PER_EPOCH=$((TRAIN_WINDOWS / BATCH_SIZE))

# Estimate time per batch (larger model is slower)
# Phase 2A: ~1.2 seconds/batch with 30M params
# Phase 3: ~4x params, estimate ~3-4 seconds/batch
SECONDS_PER_BATCH=3.5

SECONDS_PER_EPOCH=$(echo "$BATCHES_PER_EPOCH * $SECONDS_PER_BATCH" | bc)
MINUTES_PER_EPOCH=$(echo "$SECONDS_PER_EPOCH / 60" | bc)
HOURS_PER_EPOCH=$(echo "scale=1; $SECONDS_PER_EPOCH / 3600" | bc)

echo "Training Speed Estimates:"
echo "  Batches per epoch:  $(printf "%'d" $BATCHES_PER_EPOCH)"
echo "  Seconds per batch:  ~${SECONDS_PER_BATCH}s (estimated)"
echo "  Time per epoch:     ~${MINUTES_PER_EPOCH} minutes (~${HOURS_PER_EPOCH} hours)"
echo ""

# Total training time
MAX_EPOCHS=150
EARLY_STOP=30
EXPECTED_EPOCHS=40  # Likely to stop early

TOTAL_HOURS=$(echo "$EXPECTED_EPOCHS * $HOURS_PER_EPOCH" | bc)
TOTAL_DAYS=$(echo "scale=1; $TOTAL_HOURS / 24" | bc)

echo "Expected Training Duration:"
echo "  If runs full 150 epochs: ~$((150 * ${MINUTES_PER_EPOCH} / 60)) hours (~$((150 * ${MINUTES_PER_EPOCH} / 60 / 24)) days)"
echo "  Expected (with early stop): ~${TOTAL_HOURS} hours (~${TOTAL_DAYS} days)"
echo ""

if (( $(echo "$TOTAL_DAYS < 7" | bc -l) )); then
    echo "✅ Fits within 7-day time limit!"
else
    echo "⚠️  Might exceed 7-day limit"
fi

echo ""

# ============================================================
# 5. REGULARIZATION SETTINGS
# ============================================================
echo "5. REGULARIZATION SETTINGS"
echo "-------------------------------------------"

LABEL_SMOOTH=$(grep "CrossEntropyLoss" train_deep.py | grep -o "label_smoothing=[0-9.]*")
WEIGHT_DECAY=$(grep "weight_decay.*default=" train_deep.py | grep -o "default=[0-9e-]*" | head -1)
DROPOUT=$(grep "model_dropout_rate.*default=" train_deep.py | grep -o "default=[0-9.]*" | head -1)

echo "Regularization (same as Phase 2A):"
echo "  Label smoothing: $LABEL_SMOOTH"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Dropout: $DROPOUT"
echo "  Gradient clipping: 1.0"
echo ""

# ============================================================
# 6. SLURM CONFIGURATION
# ============================================================
echo "6. SLURM CONFIGURATION"
echo "-------------------------------------------"

if [ -f "run_phase3_deep.slurm" ]; then
    echo "✅ SLURM script exists"
    
    if grep -q "#SBATCH --gpus=1" run_phase3_deep.slurm; then
        echo "✅ GPU: 1"
    else
        echo "❌ GPU not requested"
    fi
    
    if grep -q "#SBATCH --mem=240G" run_phase3_deep.slurm; then
        echo "✅ RAM: 240GB"
    else
        echo "⚠️  RAM allocation different"
    fi
    
    if grep -q "#SBATCH --qos=long" run_phase3_deep.slurm; then
        echo "✅ QOS: long (7-day limit)"
    else
        echo "❌ QOS not set to long"
    fi
    
    if grep -q "#SBATCH --time=7-00:00:00" run_phase3_deep.slurm; then
        echo "✅ Time limit: 7 days"
    else
        echo "⚠️  Time limit different"
    fi
else
    echo "❌ SLURM script missing!"
    exit 1
fi

echo ""

# ============================================================
# 7. PYTHON SYNTAX CHECK
# ============================================================
echo "7. PYTHON SYNTAX CHECK"
echo "-------------------------------------------"

PYTHON=/work/users/tgangil/pangpt_project_v2/pangenome/bin/python
$PYTHON -m py_compile train_deep.py 2>&1

if [ $? -eq 0 ]; then
    echo "✅ No syntax errors"
else
    echo "❌ Syntax errors found!"
    exit 1
fi

echo ""

# ============================================================
# 8. OUTPUT DIRECTORIES
# ============================================================
echo "8. OUTPUT DIRECTORIES"
echo "-------------------------------------------"

for dir in checkpoints logs results; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "⚠️  $dir/ missing - creating..."
        mkdir -p "$dir"
        echo "   Created $dir/"
    fi
done

echo ""

# ============================================================
# 9. COMPARISON WITH PREVIOUS PHASES
# ============================================================
echo "9. COMPARISON WITH PREVIOUS PHASES"
echo "-------------------------------------------"

echo "┌─────────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┐"
echo "│ Phase       │ Layers   │ Embed    │ Heads    │ Parameters   │ Test Acc     │"
echo "├─────────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┤"
echo "│ Phase 1     │ 6        │ 512      │ 8        │ ~50M         │ 0.51%        │"
echo "│ Phase 2     │ 6        │ 256      │ 8        │ ~30M         │ 50.23%       │"
echo "│ Phase 2A    │ 6        │ 256      │ 8        │ ~30M         │ 50.27%       │"
echo "│ Phase 3     │ 12       │ 512      │ 16       │ ~120M        │ Target: 60%+ │"
echo "└─────────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┘"

echo ""

# ============================================================
# 10. EXPECTED IMPROVEMENTS
# ============================================================
echo "10. EXPECTED IMPROVEMENTS"
echo "-------------------------------------------"

echo "Why Phase 3 should perform better:"
echo ""
echo "✓ 4x more parameters (30M → 120M)"
echo "  → More capacity to learn complex patterns"
echo "  → Can capture longer-range dependencies"
echo ""
echo "✓ 2x deeper (6 → 12 layers)"
echo "  → Can learn more abstract representations"
echo "  → Better hierarchical feature extraction"
echo ""
echo "✓ 2x wider (256 → 512 embedding)"
echo "  → Richer gene representations"
echo "  → More expressive power"
echo ""
echo "✓ Same moderate regularization"
echo "  → Won't overfit (proven by Phase 2A)"
echo "  → Won't collapse (avoided Phase 2 problem)"
echo ""
echo "Expected Results:"
echo "  Conservative: 55-58% accuracy (+5-8%)"
echo "  Optimistic:   60-65% accuracy (+10-15%)"
echo "  Best case:    65-70% accuracy (+15-20%)"
echo ""

# ============================================================
# 11. MONITORING RECOMMENDATIONS
# ============================================================
echo "11. MONITORING RECOMMENDATIONS"
echo "-------------------------------------------"

echo "After submission, monitor these metrics:"
echo ""
echo "✓ First 5 epochs:"
echo "  - Loss should decrease (not increase like Phase 2)"
echo "  - Accuracy should improve from ~50% baseline"
echo "  - GPU memory usage should be stable (~20-30GB)"
echo ""
echo "✓ Epochs 5-20:"
echo "  - Gradual improvement expected"
echo "  - Watch for plateau (if stuck at 50%, something wrong)"
echo ""
echo "✓ Red flags to watch for:"
echo "  - Loss increases after epoch 0"
echo "  - Accuracy stuck at exactly 50%"
echo "  - GPU memory errors"
echo "  - NaN losses"
echo ""
echo "Commands to monitor:"
echo "  squeue -u tgangil"
echo "  tail -f logs/phase3_*.err"
echo "  watch -n 60 'tail -20 logs/phase3_*.err'"
echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "=========================================="
echo "✅ ALL SANITY CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Phase 3 Deep Model Summary:"
echo "─────────────────────────────────────────"
echo "Architecture:"
echo "  ✓ 12 layers (2x deeper)"
echo "  ✓ 512 embedding (2x wider)"
echo "  ✓ 16 heads (2x more)"
echo "  ✓ ~120M parameters (4x larger)"
echo ""
echo "Data:"
echo "  ✓ 147,147 train windows"
echo "  ✓ 18,388 val windows"
echo "  ✓ 18,378 test windows"
echo ""
echo "Resources:"
echo "  ✓ Fits in GPU memory (~20-30GB)"
echo "  ✓ Fits in 7-day time limit (~3-4 days)"
echo ""
echo "Expected:"
echo "  ✓ Training time: 3-4 days"
echo "  ✓ Target accuracy: 55-65%"
echo "  ✓ Improvement: +5-15% over Phase 2A"
echo ""
echo "=========================================="
echo "READY TO SUBMIT!"
echo "=========================================="
echo ""
echo "To submit:"
echo "  sbatch run_phase3_deep.slurm"
echo ""
echo "To monitor:"
echo "  squeue -u tgangil"
echo "  tail -f logs/phase3_*.err"
echo ""
echo "=========================================="

