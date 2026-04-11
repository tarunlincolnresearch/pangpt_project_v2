-#!/bin/bash

echo "================================================================================================"
echo "VOCABULARY REDUCTION EXPERIMENT"
echo "================================================================================================"

# Step 1: Analyze vocabulary
echo ""
echo "Step 1: Analyzing vocabulary distribution..."
echo "------------------------------------------------------------------------------------------------"
cd scripts
python analyze_vocabulary.py
cd..

echo ""
read -p "Review the analysis above. Press Enter to continue with vocabulary reduction..."

# Step 2: Create reduced vocabulary datasets
echo ""
echo "Step 2: Creating reduced vocabulary datasets..."
echo "------------------------------------------------------------------------------------------------"

# Test multiple thresholds
for THRESHOLD in 5 10 15 20; do
    echo ""
    echo "Creating dataset with threshold ≥${THRESHOLD}..."
    python scripts/create_reduced_vocab.py \
        --input_file../james_exact_approach/data/all_genomes.txt \
        --threshold $THRESHOLD \
        --output_dir data
    echo "✓ Threshold ${THRESHOLD} complete"
done

echo ""
echo "================================================================================================"
echo "✓ VOCABULARY REDUCTION COMPLETE!"
echo "================================================================================================"
echo ""
echo "Created datasets:"
ls -lh data/all_genomes_vocab*.txt
echo ""
echo "Vocabulary info:"
ls -lh data/vocab_info_*.txt
echo ""
echo "================================================================================================"
echo "Next steps:"
echo "  1. Review vocabulary statistics in data/vocab_info_*.txt"
echo "  2. Choose optimal threshold (recommended: 10)"
echo "  3. Train model: python scripts/train_reduced_vocab.py --input_file data/all_genomes_vocab10.txt"
echo "================================================================================================"

