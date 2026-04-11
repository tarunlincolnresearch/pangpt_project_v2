# James Improved - Vocabulary Reduction

This folder contains the improved version of the James model with vocabulary reduction.

## Approach

**Problem:** Original vocabulary of 70,000 genes is too large
- Many genes appear only 1-2 times (can't be learned)
- Model predicts frequent genes, ignores rare ones
- Accuracy plateaus at 65%

**Solution:** Reduce vocabulary by mapping rare genes to `<RARE>` token
- Keep only genes appearing ≥N times (threshold)
- Map infrequent genes to special `<RARE>` token
- Expected improvement: +10-15% accuracy

## Directory Structure

