#!/bin/bash

echo "================================================================================================"
echo "SAVING ALL PANPROMPT OUTPUTS FOR SUPERVISOR"
echo "================================================================================================"

# Create output directory
mkdir -p panprompt_outputs

# Step 1: Save sequential generation outputs
echo ""
echo "Step 1: Saving sequential generation outputs (panPrompt.py)..."
echo "------------------------------------------------------------------------------------------------"
bash save_panprompt_generation.sh

# Step 2: Save detailed predictions for 5 genomes
echo ""
echo "Step 2: Saving detailed predictions with probabilities for 5 genomes..."
echo "------------------------------------------------------------------------------------------------"
python visualizations/show_top_predictions_5genomes.py

# Step 3: Copy the earlier random genome analysis
echo ""
echo "Step 3: Copying earlier analysis results..."
echo "------------------------------------------------------------------------------------------------"
cp visualizations/panprompt_results.txt panprompt_outputs/random_genome_analysis.txt
echo "✓ Copied to: panprompt_outputs/random_genome_analysis.txt"

# Create summary document
echo ""
echo "Step 4: Creating summary document..."
echo "------------------------------------------------------------------------------------------------"

cat > panprompt_outputs/README.txt << 'EOF'
================================================================================================
PANPROMPT OUTPUTS - SUMMARY
================================================================================================

This directory contains all outputs from the panPrompt analysis of the James model.

FILES:
------------------------------------------------------------------------------------------------

1. generation_test1.txt
   - Sequential generation of 20 genes
   - Settings: repetition_penalty=1.2, temperature=1.0
   - Shows how model continues a genome sequence

2. generation_test2.txt
   - Sequential generation with higher repetition penalty (2.0)
   - Attempts to reduce repetition loops

3. generation_test3.txt
   - Sequential generation with higher temperature (1.5)
   - More randomness in predictions

4. detailed_predictions_5genomes.txt
   - Detailed analysis of 5 random test genomes
   - Shows top 50 predictions with probabilities for each
   - Includes actual gene rank and probability distribution

5. random_genome_analysis.txt
   - Earlier analysis showing top-k accuracy metrics
   - Summary statistics across 5 genomes

================================================================================================
KEY FINDINGS:
------------------------------------------------------------------------------------------------

Model Performance:
- Vocabulary size: 70,000 genes
- Test accuracy: 65.41% (token-level)
- Sequential generation shows repetition patterns
- Top predictions are dominated by most frequent genes

Challenges:
- Very large vocabulary makes exact prediction difficult
- Model tends to predict common genes
- Actual genes are often rare (appear <0.04% of time)
- Sequential generation can get stuck in loops

================================================================================================
EOF

echo "✓ Created: panprompt_outputs/README.txt"

# List all generated files
echo ""
echo "================================================================================================"
echo "✓ ALL OUTPUTS SAVED!"
echo "================================================================================================"
echo ""
echo "Generated files in panprompt_outputs/:"
ls -lh panprompt_outputs/
echo ""
echo "================================================================================================"
