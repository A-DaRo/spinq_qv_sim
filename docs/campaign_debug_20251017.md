# Campaign System Debugging - October 17, 2025

## Issues Fixed

### 1. Import Error: StatevectorSimulator Not Found
**Fixed**: Changed campaign_executor to use existing QV experiment runner instead of reimplementing simulation logic.

### 2. Plotting Crashes on Failed Results  
**Fixed**: Added None checks in all 5 plotting functions to handle failed configurations gracefully.

## Quick Test

```bash
python examples/run_parameter_campaign.py \
    --base-config examples/configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --output test_debug
```

Should complete in ~2 minutes without errors.
