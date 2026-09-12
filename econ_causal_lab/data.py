"""Verified downloads from the researchers' public data archive.

The raw data are not distributed with this MIT-licensed software. Their terms
are described by the source at https://users.nber.org/~rdehejia/nswdata2.html.
"""

import hashlib
import io
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


FEATURES = ["age","education","black","hispanic","married","nodegree","re74","re75"]
COLUMNS = ["treatment",*FEATURES,"re78"]
SOURCES = [
    dict(name="nswre74_treated",rows=185,treatment=1,
         url="https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
         sha256="e7b742fe0ff07a0f45e129b4ff108bb9611cd83d53604732c48a8a0a3e20eda3"),
    dict(name="nswre74_control",rows=260,treatment=0,
         url="https://users.nber.org/~rdehejia/data/nswre74_control.txt",
         sha256="a1364cea459d953dc691a667d99194b4ad335d6d550354fe23a5d2dc58d729b5"),
    dict(name="cps_controls",rows=15992,treatment=0,
         url="https://users.nber.org/~rdehejia/data/cps_controls.txt",
         sha256="dce640066f8bcdf447864b158b2cb0390bd3f8d6cbe3a27576a06926477d51f8"),
]


def verified_array(content,source):
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError(f"{source['name']}: SHA-256 mismatch; source changed or cache is damaged")
    array = np.loadtxt(io.BytesIO(content))
    if array.shape != (source["rows"],10) or not np.isfinite(array).all():
        raise ValueError(f"{source['name']}: unexpected data shape or nonfinite values")
    if not np.all(array[:,0] == source["treatment"]):
        raise ValueError(f"{source['name']}: unexpected treatment assignment")
    return array


def load_benchmark(cache="data/raw",offline=False):
    cache = Path(cache)
    arrays = []
    for source in SOURCES:
        path = cache/(source["name"]+".txt")
        if path.exists():
            content = path.read_bytes()
            array = verified_array(content,source)
        else:
            if offline:
                raise FileNotFoundError(f"Missing cache: {path}. Run without --offline once to download.")
            request = urllib.request.Request(source["url"],headers={"User-Agent":"econ-causal-lab/0.1"})
            with urllib.request.urlopen(request,timeout=60) as response:
                content = response.read()
            array = verified_array(content,source)
            cache.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
        arrays.append(array)
    treated,random_controls,cps_controls = arrays
    return {
        "experimental":pd.DataFrame(np.concatenate((treated,random_controls)),columns=COLUMNS),
        "observational":pd.DataFrame(np.concatenate((treated,cps_controls)),columns=COLUMNS),
    }


def load_csv(path,outcome,treatment,features):
    if not features or len(set(features)) != len(features) or outcome == treatment:
        raise ValueError("supply unique feature columns and distinct outcome/treatment")
    if outcome in features or treatment in features:
        raise ValueError("outcome and treatment cannot be included as covariates")
    frame = pd.read_csv(path,encoding="utf-8-sig")
    columns = [outcome,treatment,*features]
    missing = set(columns)-set(frame.columns)
    if missing:
        raise ValueError("missing columns: "+", ".join(sorted(missing)))
    try:
        selected = frame[columns].apply(pd.to_numeric,errors="raise")
    except (ValueError,TypeError) as exc:
        raise ValueError("the selected columns must be numeric; encode categories explicitly") from exc
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("selected data contain missing or nonfinite values; define a cleaning strategy first")
    return selected
