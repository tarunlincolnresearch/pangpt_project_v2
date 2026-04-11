#!/bin/bash

echo "================================================================================================"
echo "JAMES MODEL - COMPREHENSIVE ANALYSIS"
echo "================================================================================================"

# Set paths - CORRECTED
LOG_DIR="logs"
MODEL_PATH="checkpoints/james_model.pth"
TOKENIZER_PATH="tokenizer.json"
TEST_FILE="data/all_genomes.txt"
VIZ_DIR="visualizations"

# Step 1: Plot training metrics
echo ""
echo "Step 1: Generating training metric visualizations..."
echo "------------------------------------------------------------------------------------------------"
python visualizations/plot_training_metrics.py \
    --log_dir $LOG_DIR \
    --output_dir $VIZ_DIR

# Step 2: Run pan-prompting on random test genomes
echo ""
echo "Step 2: Running pan-prompting on 5 random test genomes..."
echo "------------------------------------------------------------------------------------------------"
python visualizations/analyze_random_genomes.py \
    --model_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --test_file $TEST_FILE \
    --num_genomes 5 \
    --top_k 50 \
    --output $VIZ_DIR/panprompt_results.txt \
    --device cpu

echo ""
echo "================================================================================================"
echo "✓ ANALYSIS COMPLETE!"
echo "================================================================================================"
echo "Generated files:"
echo "  - $VIZ_DIR/*.png (training metric plots)"
echo "  - $VIZ_DIR/panprompt_results.txt (prediction results)"
echo ""
echo "View results:"
echo "  cat $VIZ_DIR/panprompt_results.txt"
echo "  ls -lh $VIZ_DIR/"
echo "================================================================================================"