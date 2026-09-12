"""Generate a shareable synthetic dataset with a known constant ATT of 2000.

This is simulated data, not an extract of the NSW or CPS microdata.
"""
from pathlib import Path

import pandas as pd

from econ_causal_lab.simulation import generate


def main():
    x,y,d,truth = generate(n=600,seed=42,scenario="good_overlap")
    frame = pd.DataFrame(x,columns=[f"x{i}" for i in range(x.shape[1])])
    frame["treatment"],frame["outcome"] = d,y
    target = Path(__file__).with_name("synthetic.csv")
    frame.to_csv(target,index=False,float_format="%.10g")
    print(f"Wrote {target.name}; simulated population ATT = {truth}")


if __name__ == "__main__":
    main()
