#!/bin/bash

echo "=========================================="
echo "DATASET USAGE ANALYSIS"
echo "=========================================="
echo ""

# ============================================================
# 1. RAW GENOME DATA
# ============================================================
echo "1. ORIGINAL GENOME DATA"
echo "-------------------------------------------"

# Check the original gene order file
if [ -f "data/cleaned_gene_orders.json" ]; then
    echo "Original data file: data/cleaned_gene_orders.json"
    FILE_SIZE=$(du -h data/cleaned_gene_orders.json | cut -f1)
    echo "File size: $FILE_SIZE"
    
    # Count total genomes
    TOTAL_GENOMES=$(python3 -c "import json; data=json.load(open('data/cleaned_gene_orders.json')); print(len(data))")
    echo "Total genomes in dataset: $TOTAL_GENOMES"
    echo ""
fi

# ============================================================
# 2. GENOME-LEVEL SPLIT (Before Windowing)
# ============================================================
echo "2. GENOME-LEVEL SPLIT (80-10-10)"
echo "-------------------------------------------"

# Check phase1 summary
if [ -f "data/phase1/phase1_summary.json" ]; then
    echo "Reading split information..."
    
    TRAIN_GENOMES=$(python3 -c "import json; d=json.load(open('data/phase1/phase1_summary.json')); print(d['win256']['train']['genomes'])")
    VAL_GENOMES=$(python3 -c "import json; d=json.load(open('data/phase1/phase1_summary.json')); print(d['win256']['val']['genomes'])")
    TEST_GENOMES=$(python3 -c "import json; d=json.load(open('data/phase1/phase1_summary.json')); print(d['win256']['test']['genomes'])")
    
    echo "Train genomes: $TRAIN_GENOMES"
    echo "Val genomes:   $VAL_GENOMES"
    echo "Test genomes:  $TEST_GENOMES"
    echo "Total:         $((TRAIN_GENOMES + VAL_GENOMES + TEST_GENOMES))"
    echo ""
fi

# ============================================================
# 3. WINDOWING PROCESS (Window Size = 256)
# ============================================================
echo "3. WINDOWING PROCESS (Window Size = 256)"
echo "-------------------------------------------"

TRAIN_FILE="data/phase1/win256/train_windows.txt"
VAL_FILE="data/phase1/win256/val_windows.txt"
TEST_FILE="data/phase1/win256/test_windows.txt"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_WINDOWS=$(wc -l < "$TRAIN_FILE")
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    echo "Train windows file: $TRAIN_FILE"
    echo "  Windows created: $TRAIN_WINDOWS"
    echo "  File size: $TRAIN_SIZE"
    
    # Sample first window to show structure
    echo "  Sample window (first line):"
    head -1 "$TRAIN_FILE" | cut -d' ' -f1-10 | tr ' ' '\n' | nl
    echo "... (256 genes total per window)"
    echo ""
fi

if [ -f "$VAL_FILE" ]; then
    VAL_WINDOWS=$(wc -l < "$VAL_FILE")
    VAL_SIZE=$(du -h "$VAL_FILE" | cut -f1)
    echo "Validation windows file: $VAL_FILE"
    echo "  Windows created: $VAL_WINDOWS"
    echo "  File size: $VAL_SIZE"
    echo ""
fi

if [ -f "$TEST_FILE" ]; then
    TEST_WINDOWS=$(wc -l < "$TEST_FILE")
    TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
    echo "Test windows file: $TEST_FILE"
    echo "  Windows created: $TEST_WINDOWS"
    echo "  File size: $TEST_SIZE"
    echo ""
fi

TOTAL_WINDOWS=$((TRAIN_WINDOWS + VAL_WINDOWS + TEST_WINDOWS))
echo "Total windows across all splits: $TOTAL_WINDOWS"
echo ""

# ============================================================
# 4. HOW WINDOWING WORKS
# ============================================================
echo "4. HOW WINDOWING WORKS"
echo "-------------------------------------------"

echo "Example: If a genome has 1000 genes and window size = 256:"
echo ""
echo "  Window 1:  genes[0:256]     → positions 1-256"
echo "  Window 2:  genes[256:512]   → positions 257-512"
echo "  Window 3:  genes[512:768]   → positions 513-768"
echo "  Window 4:  genes[768:1000]  → positions 769-1000 (remainder, padded)"
echo ""
echo "  Result: 4 windows from 1 genome"
echo ""
echo "Key points:"
echo "  ✓ Each genome is split into multiple windows"
echo "  ✓ Windows don't overlap (shift = window_size)"
echo "  ✓ Last window may be shorter (gets padded)"
echo "  ✓ Gene order is preserved within each window"
echo ""

# ============================================================
# 5. CALCULATE AVERAGE WINDOWS PER GENOME
# ============================================================
echo "5. WINDOWS PER GENOME STATISTICS"
echo "-------------------------------------------"

if [ ! -z "$TRAIN_GENOMES" ] && [ ! -z "$TRAIN_WINDOWS" ]; then
    AVG_WINDOWS_TRAIN=$(echo "scale=2; $TRAIN_WINDOWS / $TRAIN_GENOMES" | python3 -c "print(round($TRAIN_WINDOWS / $TRAIN_GENOMES, 2))")
    echo "Train set:"
    echo "  $TRAIN_GENOMES genomes → $TRAIN_WINDOWS windows"
    echo "  Average: ~$AVG_WINDOWS_TRAIN windows per genome"
    echo ""
fi

if [ ! -z "$VAL_GENOMES" ] && [ ! -z "$VAL_WINDOWS" ]; then
    AVG_WINDOWS_VAL=$(echo "scale=2; $VAL_WINDOWS / $VAL_GENOMES" | python3 -c "print(round($VAL_WINDOWS / $VAL_GENOMES, 2))")
    echo "Validation set:"
    echo "  $VAL_GENOMES genomes → $VAL_WINDOWS windows"
    echo "  Average: ~$AVG_WINDOWS_VAL windows per genome"
    echo ""
fi

if [ ! -z "$TEST_GENOMES" ] && [ ! -z "$TEST_WINDOWS" ]; then
    AVG_WINDOWS_TEST=$(echo "scale=2; $TEST_WINDOWS / $TEST_GENOMES" | python3 -c "print(round($TEST_WINDOWS / $TEST_GENOMES, 2))")
    echo "Test set:"
    echo "  $TEST_GENOMES genomes → $TEST_WINDOWS windows"
    echo "  Average: ~$AVG_WINDOWS_TEST windows per genome"
    echo ""
fi

# ============================================================
# 6. TOTAL GENES IN DATASET
# ============================================================
echo "6. TOTAL GENES IN DATASET"
echo "-------------------------------------------"

# Count total genes in training windows
if [ -f "$TRAIN_FILE" ]; then
    echo "Counting genes in training data (this may take a moment)..."
    TOTAL_GENES=$(head -1000 "$TRAIN_FILE" | wc -w)
    SAMPLE_WINDOWS=1000
    AVG_GENES_PER_WINDOW=$(echo "scale=0; $TOTAL_GENES / $SAMPLE_WINDOWS" | python3 -c "print(int($TOTAL_GENES / $SAMPLE_WINDOWS))")
    ESTIMATED_TOTAL_GENES=$(echo "$AVG_GENES_PER_WINDOW * $TRAIN_WINDOWS" | python3 -c "print($AVG_GENES_PER_WINDOW * $TRAIN_WINDOWS)")
    
    echo "Sample analysis (first 1000 windows):"
    echo "  Total genes: $TOTAL_GENES"
    echo "  Average genes per window: $AVG_GENES_PER_WINDOW"
    echo ""
    echo "Estimated total genes in training set:"
    echo "  $TRAIN_WINDOWS windows × $AVG_GENES_PER_WINDOW genes/window"
    echo "  ≈ $(printf "%'d" $ESTIMATED_TOTAL_GENES) genes"
    echo ""
fi

# ============================================================
# 7. WHAT MODEL ACTUALLY SEES
# ============================================================
echo "7. WHAT THE MODEL SEES DURING TRAINING"
echo "-------------------------------------------"

BATCH_SIZE=4
EPOCHS=150
BATCHES_PER_EPOCH=$(echo "$TRAIN_WINDOWS / $BATCH_SIZE" | python3 -c "print(int($TRAIN_WINDOWS / $BATCH_SIZE))")
TOTAL_BATCHES=$(echo "$BATCHES_PER_EPOCH * $EPOCHS" | python3 -c "print($BATCHES_PER_EPOCH * $EPOCHS)")

echo "Training configuration:"
echo "  Batch size: $BATCH_SIZE windows"
echo "  Training windows: $TRAIN_WINDOWS"
echo "  Batches per epoch: $BATCHES_PER_EPOCH"
echo "  Max epochs: $EPOCHS"
echo ""
echo "Total training iterations:"
echo "  $BATCHES_PER_EPOCH batches/epoch × $EPOCHS epochs"
echo "  = $(printf "%'d" $TOTAL_BATCHES) total batches"
echo ""
echo "Each batch contains:"
echo "  $BATCH_SIZE windows × 256 genes/window = $(($BATCH_SIZE * 256)) genes"
echo ""

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
echo "8. COMPLETE DATASET SUMMARY"
echo "-------------------------------------------"

echo "┌─────────────────────┬───────────┬─────────────┬──────────────┐"
echo "│ Split               │ Genomes   │ Windows     │ Genes (est.) │"
echo "├─────────────────────┼───────────┼─────────────┼──────────────┤"
printf "│ %-19s │ %9s │ %11s │ %12s │\n" "Training" "$TRAIN_GENOMES" "$(printf "%'d" $TRAIN_WINDOWS)" "$(printf "%'d" $(($TRAIN_WINDOWS * 256)))"
printf "│ %-19s │ %9s │ %11s │ %12s │\n" "Validation" "$VAL_GENOMES" "$(printf "%'d" $VAL_WINDOWS)" "$(printf "%'d" $(($VAL_WINDOWS * 256)))"
printf "│ %-19s │ %9s │ %11s │ %12s │\n" "Test" "$TEST_GENOMES" "$(printf "%'d" $TEST_WINDOWS)" "$(printf "%'d" $(($TEST_WINDOWS * 256)))"
echo "├─────────────────────┼───────────┼─────────────┼──────────────┤"
printf "│ %-19s │ %9s │ %11s │ %12s │\n" "TOTAL" "$((TRAIN_GENOMES + VAL_GENOMES + TEST_GENOMES))" "$(printf "%'d" $TOTAL_WINDOWS)" "$(printf "%'d" $(($TOTAL_WINDOWS * 256)))"
echo "└─────────────────────┴───────────┴─────────────┴──────────────┘"

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="

