#!/bin/bash

echo "================================================================================================"
echo "SAVING PANPROMPT SEQUENTIAL GENERATION OUTPUTS"
echo "================================================================================================"

mkdir -p panprompt_outputs

# First, extract the actual continuation from the genome
echo "Extracting actual genome sequence for comparison..."

# Get the full genome (first line)
FULL_GENOME=$(head -1 data/all_genomes.txt)

# Get first 100 genes (prompt)
PROMPT=$(echo "$FULL_GENOME" | awk '{for(i=1;i<=100;i++) printf "%s ", $i}')

# Get next 20 genes (actual continuation)
ACTUAL=$(echo "$FULL_GENOME" | awk '{for(i=101;i<=120;i++) printf "%s ", $i}')

echo "$PROMPT" > test_prompt.txt

# Test 1: Standard generation
echo ""
echo "Test 1: Standard generation (20 genes, repetition_penalty=1.2)"
echo "------------------------------------------------------------------------------------------------"

PREDICTED=$(python panPrompt.py \
    --model_path checkpoints/james_model.pth \
    --model_type transformer \
    --tokenizer_path tokenizer.json \
    --prompt_file test_prompt.txt \
    --num_tokens 20 \
    --temperature 1.0 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 1.2 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 4096 \
    --dropout_rate 0.2 \
    --device cpu 2>/dev/null | grep "Predicted text:" | sed 's/Predicted text: //')

# Extract only the NEW genes (remove the prompt)
PREDICTED_NEW=$(echo "$PREDICTED" | awk '{for(i=101;i<=120;i++) printf "%s ", $i}')

cat > panprompt_outputs/generation_test1.txt << EOF
================================================================================================
GENERATION TEST 1: Standard Settings
================================================================================================

Settings:
  - Number of tokens to generate: 20
  - Temperature: 1.0
  - Top-k: 50
  - Top-p: 0.95
  - Repetition penalty: 1.2

================================================================================================
INPUT PROMPT (First 100 genes):
================================================================================================
$PROMPT

================================================================================================
PREDICTED CONTINUATION (Next 20 genes):
================================================================================================
$PREDICTED_NEW

================================================================================================
ACTUAL CONTINUATION (Next 20 genes from original genome):
================================================================================================
$ACTUAL

================================================================================================
COMPARISON:
================================================================================================

Predicted genes: $(echo "$PREDICTED_NEW" | wc -w) genes
Actual genes:    $(echo "$ACTUAL" | wc -w) genes

Predicted: $PREDICTED_NEW

Actual:    $ACTUAL

================================================================================================
ANALYSIS:
================================================================================================

EOF

# Calculate how many genes match
MATCH_COUNT=0
for i in {1..20}; do
    PRED_GENE=$(echo "$PREDICTED_NEW" | awk -v i=$i '{print $i}')
    ACT_GENE=$(echo "$ACTUAL" | awk -v i=$i '{print $i}')
    if [ "$PRED_GENE" == "$ACT_GENE" ]; then
        MATCH_COUNT=$((MATCH_COUNT + 1))
        echo "Position $i: ✓ MATCH - $PRED_GENE" >> panprompt_outputs/generation_test1.txt
    else
        echo "Position $i: ✗ DIFF - Predicted: $PRED_GENE | Actual: $ACT_GENE" >> panprompt_outputs/generation_test1.txt
    fi
done

echo "" >> panprompt_outputs/generation_test1.txt
echo "Exact matches: $MATCH_COUNT / 20 ($(echo "scale=1; $MATCH_COUNT * 100 / 20" | bc)%)" >> panprompt_outputs/generation_test1.txt
echo "================================================================================================" >> panprompt_outputs/generation_test1.txt

echo "✓ Saved to: panprompt_outputs/generation_test1.txt"

# Test 2: Higher repetition penalty
echo ""
echo "Test 2: Higher repetition penalty (20 genes, repetition_penalty=2.0)"
echo "------------------------------------------------------------------------------------------------"

PREDICTED=$(python panPrompt.py \
    --model_path checkpoints/james_model.pth \
    --model_type transformer \
    --tokenizer_path tokenizer.json \
    --prompt_file test_prompt.txt \
    --num_tokens 20 \
    --temperature 1.0 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 2.0 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 4096 \
    --dropout_rate 0.2 \
    --device cpu 2>/dev/null | grep "Predicted text:" | sed 's/Predicted text: //')

PREDICTED_NEW=$(echo "$PREDICTED" | awk '{for(i=101;i<=120;i++) printf "%s ", $i}')

cat > panprompt_outputs/generation_test2.txt << EOF
================================================================================================
GENERATION TEST 2: Higher Repetition Penalty
================================================================================================

Settings:
  - Number of tokens to generate: 20
  - Temperature: 1.0
  - Top-k: 50
  - Top-p: 0.95
  - Repetition penalty: 2.0 (increased to reduce loops)

================================================================================================
INPUT PROMPT (First 100 genes):
================================================================================================
$PROMPT

================================================================================================
PREDICTED CONTINUATION (Next 20 genes):
================================================================================================
$PREDICTED_NEW

================================================================================================
ACTUAL CONTINUATION (Next 20 genes from original genome):
================================================================================================
$ACTUAL

================================================================================================
COMPARISON:
================================================================================================

Predicted genes: $(echo "$PREDICTED_NEW" | wc -w) genes
Actual genes:    $(echo "$ACTUAL" | wc -w) genes

Predicted: $PREDICTED_NEW

Actual:    $ACTUAL

================================================================================================
ANALYSIS:
================================================================================================

EOF

MATCH_COUNT=0
for i in {1..20}; do
    PRED_GENE=$(echo "$PREDICTED_NEW" | awk -v i=$i '{print $i}')
    ACT_GENE=$(echo "$ACTUAL" | awk -v i=$i '{print $i}')
    if [ "$PRED_GENE" == "$ACT_GENE" ]; then
        MATCH_COUNT=$((MATCH_COUNT + 1))
        echo "Position $i: ✓ MATCH - $PRED_GENE" >> panprompt_outputs/generation_test2.txt
    else
        echo "Position $i: ✗ DIFF - Predicted: $PRED_GENE | Actual: $ACT_GENE" >> panprompt_outputs/generation_test2.txt
    fi
done

echo "" >> panprompt_outputs/generation_test2.txt
echo "Exact matches: $MATCH_COUNT / 20 ($(echo "scale=1; $MATCH_COUNT * 100 / 20" | bc)%)" >> panprompt_outputs/generation_test2.txt
echo "================================================================================================" >> panprompt_outputs/generation_test2.txt

echo "✓ Saved to: panprompt_outputs/generation_test2.txt"

# Test 3: Higher temperature
echo ""
echo "Test 3: Higher temperature (20 genes, temperature=1.5)"
echo "------------------------------------------------------------------------------------------------"

PREDICTED=$(python panPrompt.py \
    --model_path checkpoints/james_model.pth \
    --model_type transformer \
    --tokenizer_path tokenizer.json \
    --prompt_file test_prompt.txt \
    --num_tokens 20 \
    --temperature 1.5 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 1.5 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_seq_length 4096 \
    --dropout_rate 0.2 \
    --device cpu 2>/dev/null | grep "Predicted text:" | sed 's/Predicted text: //')

PREDICTED_NEW=$(echo "$PREDICTED" | awk '{for(i=101;i<=120;i++) printf "%s ", $i}')

cat > panprompt_outputs/generation_test3.txt << EOF
================================================================================================
GENERATION TEST 3: Higher Temperature
================================================================================================

Settings:
  - Number of tokens to generate: 20
  - Temperature: 1.5 (increased for more diversity)
  - Top-k: 50
  - Top-p: 0.95
  - Repetition penalty: 1.5

================================================================================================
INPUT PROMPT (First 100 genes):
================================================================================================
$PROMPT

================================================================================================
PREDICTED CONTINUATION (Next 20 genes):
================================================================================================
$PREDICTED_NEW

================================================================================================
ACTUAL CONTINUATION (Next 20 genes from original genome):
================================================================================================
$ACTUAL

================================================================================================
COMPARISON:
================================================================================================

Predicted genes: $(echo "$PREDICTED_NEW" | wc -w) genes
Actual genes:    $(echo "$ACTUAL" | wc -w) genes

Predicted: $PREDICTED_NEW

Actual:    $ACTUAL

================================================================================================
ANALYSIS:
================================================================================================

EOF

MATCH_COUNT=0
for i in {1..20}; do
    PRED_GENE=$(echo "$PREDICTED_NEW" | awk -v i=$i '{print $i}')
    ACT_GENE=$(echo "$ACTUAL" | awk -v i=$i '{print $i}')
    if [ "$PRED_GENE" == "$ACT_GENE" ]; then
        MATCH_COUNT=$((MATCH_COUNT + 1))
        echo "Position $i: ✓ MATCH - $PRED_GENE" >> panprompt_outputs/generation_test3.txt
    else
        echo "Position $i: ✗ DIFF - Predicted: $PRED_GENE | Actual: $ACT_GENE" >> panprompt_outputs/generation_test3.txt
    fi
done

echo "" >> panprompt_outputs/generation_test3.txt
echo "Exact matches: $MATCH_COUNT / 20 ($(echo "scale=1; $MATCH_COUNT * 100 / 20" | bc)%)" >> panprompt_outputs/generation_test3.txt
echo "================================================================================================" >> panprompt_outputs/generation_test3.txt

echo "✓ Saved to: panprompt_outputs/generation_test3.txt"

echo ""
echo "================================================================================================"
echo "✓ All generation tests complete with predicted vs actual comparison!"
echo "================================================================================================"