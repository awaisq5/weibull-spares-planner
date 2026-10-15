"""Generate the synthetic parts register in data/.

Scenario: one part number (a mud motor bearing pack) running in four regions.
Life in circulating hours is Weibull with shape 2.4; the scale differs by
region, with one harsher region wearing parts out sooner. Some parts are
scrapped for reasons unrelated to wear (lost in hole, handling damage, an
obsolete revision); the rest are still in service. Every value is generated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20230115)
out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)

BETA = 2.4
REGIONS = {
    "North Sea": (1650.0, 110),
    "Middle East": (1500.0, 140),
    "West Africa": (1100.0, 100),
    "North America land": (1450.0, 120),
}
LIFE_REASONS = ["worn", "washout", "cracked"]
OTHER_REASONS = ["lost in hole", "handling damage", "obsolete revision"]

rows = []
n = 0
for region, (eta, count) in REGIONS.items():
    for _ in range(count):
        n += 1
        life = eta * rng.weibull(BETA)
        age_now = rng.uniform(40, 3000)  # circulating hours accumulated so far, if never scrapped
        other_at = rng.exponential(9000)  # chance of being scrapped for a reason unrelated to wear
        end = min(life, other_at, age_now)
        if end == life:
            reason = rng.choice(LIFE_REASONS, p=[0.6, 0.25, 0.15])
        elif end == other_at:
            reason = rng.choice(OTHER_REASONS, p=[0.5, 0.3, 0.2])
        else:
            reason = ""
        rows.append(
            {
                "part_id": f"BP-{n:04d}",
                "part_number": "PN-20417",
                "description": "Bearing pack, mud motor",
                "location": region,
                "circulating_hours": round(end, 1),
                "scrap_reason": reason,
            }
        )

df = pd.DataFrame(rows)
df.to_csv(out / "parts_register.csv", index=False)

life = df.scrap_reason.isin(LIFE_REASONS)
other = df.scrap_reason.isin(OTHER_REASONS)
in_service = (~life & ~other).sum()
print(f"{len(df)} parts: {life.sum()} worn out, {other.sum()} scrapped for other reasons, {in_service} in service")
