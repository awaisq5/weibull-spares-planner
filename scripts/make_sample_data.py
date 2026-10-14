"""Generate the synthetic sample data in data/.

Scenario: bearing packs in a fleet of mud motors. True life is Weibull with
beta 2.4 and eta 1,450 operating hours; most history records are suspensions
(packs pulled at a rebuild before they failed).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20251014)
out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)

BETA, ETA = 2.4, 1450.0

# History: 220 past lives. Packs are pulled at rebuilds between 400 and 2,200 hours.
n = 220
life = ETA * rng.weibull(BETA, n)
pulled_at = rng.uniform(400, 2200, n)
failed = life <= pulled_at
hours = np.round(np.where(failed, life, pulled_at), 1)
history = pd.DataFrame(
    {
        "unit_id": [f"BP-{i:04d}" for i in range(1, n + 1)],
        "hours": hours,
        "status": np.where(failed, "failed", "suspended"),
    }
)
history.to_csv(out / "bearing_pack_history.csv", index=False)

# Installed base: 140 packs in service, younger units more common.
fleet = pd.DataFrame(
    {
        "unit_id": [f"MM-{i:03d}" for i in range(1, 141)],
        "age_hours": np.round(rng.gamma(2.2, 380, 140).clip(5, 2400), 1),
    }
)
fleet.to_csv(out / "fleet.csv", index=False)

print(f"history: {n} lives, {failed.sum()} failures, {(~failed).sum()} suspensions")
print(f"fleet: {len(fleet)} units, median age {fleet.age_hours.median():.0f} h")
