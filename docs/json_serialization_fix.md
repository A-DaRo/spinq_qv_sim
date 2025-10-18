# JSON Serialization Fix - October 17, 2025

## Issue
```
[ERROR] Failed to run F1_0p99713: Object of type bool_ is not JSON serializable
```

## Root Cause
The `run_experiment()` function returns aggregated results containing NumPy types:
- `np.bool_` (NumPy boolean)
- `np.int64`, `np.int32` (NumPy integers)
- `np.float64`, `np.float32` (NumPy floats)
- `np.ndarray` (NumPy arrays)

Python's `json.dump()` cannot serialize these types directly - it only handles native Python types (bool, int, float, list).

## Solution
Added a recursive conversion function `convert_to_json_serializable()` that:
1. Converts NumPy arrays to lists
2. Converts NumPy integers to Python int
3. Converts NumPy floats to Python float
4. Converts NumPy booleans to Python bool
5. Recursively processes dicts and lists

## Changes Made

**File**: `examples/campaign_executor.py`

**Added function** (lines 20-43):
```python
def convert_to_json_serializable(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj
```

**Updated `_save_results()` method**:
```python
# OLD:
json.dump(results, f, indent=2)

# NEW:
serializable_results = convert_to_json_serializable(results)
json.dump(serializable_results, f, indent=2)
```

**Updated `_save_campaign_summary()` method**:
```python
# OLD:
json.dump(summary, f, indent=2)

# NEW:
serializable_summary = convert_to_json_serializable(summary)
json.dump(serializable_summary, f, indent=2)
```

## Status
✅ **Fixed** - Campaign executor now handles NumPy types correctly

## Test
Your campaign should now run without JSON serialization errors. Try running it again:

```bash
python examples\run_parameter_campaign.py \
    --base-config examples\configs\production.yaml \
    --sweep-type comprehensive \
    --output campaigns\param_sweep_fixed
```

All results will be properly saved as JSON files.
