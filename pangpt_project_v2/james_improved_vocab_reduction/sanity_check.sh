#!/bin/bash

echo "================================================================================================"
echo "SANITY CHECK: REDUCED VOCABULARY TRAINING PIPELINE"
echo "================================================================================================"

PASS=0
FAIL=0

# Function to check
check() {
    if [ $? -eq 0 ]; then
        echo "  ✓ PASS: $1"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAIL: $1"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "1. Checking Input Files"
echo "------------------------------------------------------------------------------------------------"

# Check original data
test -f../james_exact_approach/data/all_genomes.txt
check "Original genome file exists"

# Check reduced vocabulary data
test -f data/all_genomes_vocab10.txt
check "Reduced vocabulary dataset exists"

# Check tracking files
test -f data/vocabulary_reduction_tracking_threshold10.txt
check "Tracking file exists"

test -f data/detailed_comparison_threshold10.txt
check "Detailed comparison file exists"

# Check file sizes
ORIGINAL_LINES=$(wc -l <../james_exact_approach/data/all_genomes.txt)
REDUCED_LINES=$(wc -l < data/all_genomes_vocab10.txt)

if [ "$ORIGINAL_LINES" -eq "$REDUCED_LINES" ]; then
    echo "  ✓ PASS: Line count matches (${ORIGINAL_LINES} genomes)"
    PASS=$((PASS + 1))
else
    echo "  ✗ FAIL: Line count mismatch (Original: ${ORIGINAL_LINES}, Reduced: ${REDUCED_LINES})"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "2. Checking Data Integrity"
echo "------------------------------------------------------------------------------------------------"

# Check for <RARE> token in reduced data
RARE_COUNT=$(grep -o "<RARE>" data/all_genomes_vocab10.txt | wc -l)
if [ "$RARE_COUNT" -gt 0 ]; then
    echo "  ✓ PASS: <RARE> token found (${RARE_COUNT} occurrences)"
    PASS=$((PASS + 1))
else
    echo "  ✗ FAIL: No <RARE> tokens found"
    FAIL=$((FAIL + 1))
fi

# Sample check: First genome
echo ""
echo "  Sample check - First genome:"
echo "  Original (first 50 genes):"
head -1../james_exact_approach/data/all_genomes.txt | awk '{for(i=1;i<=50;i++) printf "%s ", $i; print ""}'
echo ""
echo "  Reduced (first 50 genes):"
head -1 data/all_genomes_vocab10.txt | awk '{for(i=1;i<=50;i++) printf "%s ", $i; print ""}'

echo ""
echo "3. Checking Directory Structure"
echo "------------------------------------------------------------------------------------------------"

test -d checkpoints
check "checkpoints/ directory exists"

test -d logs
check "logs/ directory exists"

test -d visualizations
check "visualizations/ directory exists"

test -d scripts
check "scripts/ directory exists"

echo ""
echo "4. Checking Required Scripts"
echo "------------------------------------------------------------------------------------------------"

test -f scripts/train_reduced_vocab.py
check "Training script exists"

test -f scripts/analyze_vocabulary.py
check "Analysis script exists"

test -f scripts/create_reduced_vocab_with_tracking.py
check "Vocabulary reduction script exists"

echo ""
echo "5. Checking Vocabulary Statistics"
echo "------------------------------------------------------------------------------------------------"

# Count unique genes in reduced dataset
echo "  Counting unique genes in reduced dataset..."
UNIQUE_GENES=$(cat data/all_genomes_vocab10.txt | tr ' ' '\n' | sort -u | wc -l)
echo "  Unique genes (including <RARE>): ${UNIQUE_GENES}"

if [ "$UNIQUE_GENES" -lt 60000 ]; then
    echo "  ✓ PASS: Vocabulary reduced (${UNIQUE_GENES} < 139,249)"
    PASS=$((PASS + 1))
else
    echo "  ✗ FAIL: Vocabulary not reduced enough (${UNIQUE_GENES})"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "6. Checking Python Dependencies"
echo "------------------------------------------------------------------------------------------------"

python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" 2>/dev/null
check "PyTorch available"

python -c "import tokenizers; print('  ✓ Tokenizers available')" 2>/dev/null
check "Tokenizers library available"

python -c "import sklearn; print('  ✓ Scikit-learn available')" 2>/dev/null
check "Scikit-learn available"

python -c "import matplotlib; print('  ✓ Matplotlib available')" 2>/dev/null
check "Matplotlib available"

echo ""
echo "7. Checking GPU Availability"
echo "------------------------------------------------------------------------------------------------"

python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  GPU count:', torch.cuda.device_count())" 2>/dev/null
check "GPU check completed"

echo ""
echo "8. Testing Tokenizer Creation (Dry Run)"
echo "------------------------------------------------------------------------------------------------"

python << 'PYEOF'
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Load sample data
with open('data/all_genomes_vocab10.txt') as f:
    sample = [f.readline().strip() for _ in range(10)]

# Test tokenizer
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=70000
)

tokenizer.train_from_iterator(sample, trainer)
vocab_size = tokenizer.get_vocab_size()

print(f"  Test tokenizer vocab size: {vocab_size}")
print(f"  Sample encoding test: {len(tokenizer.encode(sample[0]).ids)} tokens")
print("  ✓ Tokenizer creation successful")
PYEOF

check "Tokenizer dry run successful"

echo ""
echo "9. Estimating Training Requirements"
echo "------------------------------------------------------------------------------------------------"

TOTAL_GENES=$(cat data/all_genomes_vocab10.txt | wc -w)
TOTAL_GENOMES=$(wc -l < data/all_genomes_vocab10.txt)

echo "  Total genomes: ${TOTAL_GENOMES}"
echo "  Total gene tokens: ${TOTAL_GENES}"
echo "  Average genes per genome: $((TOTAL_GENES / TOTAL_GENOMES))"
echo ""
echo "  Estimated requirements:"
echo "    - Training time: 4-6 hours (with GPU)"
echo "    - Memory: ~10-15 GB RAM"
echo "    - GPU memory: ~8-12 GB"
echo "    - Disk space: ~2 GB (for checkpoints)"

echo ""
echo "10. Pre-flight Check Summary"
echo "------------------------------------------------------------------------------------------------"

# Check if we have enough disk space
AVAILABLE_SPACE=$(df -h. | awk 'NR==2 {print $4}')
echo "  Available disk space: ${AVAILABLE_SPACE}"

echo ""
echo "================================================================================================"
echo "SANITY CHECK RESULTS"
echo "================================================================================================"
echo "  PASSED: ${PASS}"
echo "  FAILED: ${FAIL}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "  ✓ ALL CHECKS PASSED - Ready for training!"
    echo ""
    echo "  Next step:"
    echo "    sbatch run_training_vocab10.slurm"
    echo ""
    echo "================================================================================================"
    exit 0
else
    echo "  ✗ SOME CHECKS FAILED - Please fix issues before training"
    echo ""
    echo "================================================================================================"
    exit 1
fi

