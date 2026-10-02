"""Standards-compliant experiment JSON: unavailable numbers are null, never NaN."""
import json
import math

import numpy as np


def clean_numbers(value):
    if isinstance(value, dict):
        return {k: clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_numbers(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def dumps(value, **kwargs):
    return json.dumps(clean_numbers(value), allow_nan=False, **kwargs)
