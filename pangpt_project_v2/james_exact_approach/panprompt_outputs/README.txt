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
