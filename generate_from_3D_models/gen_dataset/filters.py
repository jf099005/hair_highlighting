"""Quality filters of the highlighting dataset generator.

A filter looks at one refined result (the flattened metrics of a contour_algo_test result.json) and says
whether it is good enough:

    @register_filter("name")
    def my_filter(m, **params) -> list[Check]

    m       dict of flat metrics, e.g. m["grid_score.mean_all"], m["grid_score.per_black_area"],
            m["strands.flip_frac"], m["shape.components"], plus derived ones (see flatten_metrics)
    Check   one tested rule: (label, value, ok); the result passes when every check of every filter is ok

A filter spec is a list of {"name": ..., "params": {...}}; every filter in it must pass. It can be given as
    - a JSON file / JSON string:  '[{"name": "grid_score_abc", "params": {"a": 0.1}}]'
    - a short string:             "grid_score_abc:a=0.1,b=0.25,c=0.1+min_retention:min=0.3"
    - an expression filter:       'expr:rule=grid_score.mean_all < 0.08 and shape.components <= 3'
      (the short form splits on "+" and ","; put an expression with commas in a JSON spec instead)
To add a new rule: write one function here, decorate it with @register_filter, use its name in the spec.
"""
import json
import math
import os
import re
from dataclasses import dataclass

FILTERS = {}


@dataclass
class Check:
    label: str        # e.g. "mean_all < 0.1"
    value: float      # the tested value
    ok: bool

    def as_dict(self):
        v = self.value
        return dict(label=self.label, value=None if v is None or (isinstance(v, float) and math.isnan(v)) else v,
                    ok=bool(self.ok))


def register_filter(name):
    def deco(fn):
        FILTERS[name] = fn
        return fn
    return deco


def _get(m, key):
    v = m.get(key)
    return float("nan") if v is None else float(v)


# ---------------------------------------------------------------- built-in filters

@register_filter("grid_score_abc")
def grid_score_abc(m, a=0.1, b=0.25, c=0.1):
    """The default rule:
        mean_all grid score                               <  a
        max(per-black-area score, per-white-area score)  <= b
        min(per-black-area score, per-white-area score)  <= c"""
    mean_all = _get(m, "grid_score.mean_all")
    pb, pw = _get(m, "grid_score.per_black_area"), _get(m, "grid_score.per_white_area")
    hi, lo = max(pb, pw), min(pb, pw)
    return [Check(f"mean_all < {a:g}", mean_all, mean_all < a),
            Check(f"max(per_black, per_white) <= {b:g}", hi, hi <= b),
            Check(f"min(per_black, per_white) <= {c:g}", lo, lo <= c)]


@register_filter("threshold")
def threshold(m, key, max=None, min=None, strict=False):
    """One metric inside [min, max] (either bound optional); strict=True uses < / >."""
    v = _get(m, key)
    out = []
    if max is not None:
        out.append(Check(f"{key} {'<' if strict else '<='} {max:g}", v, v < max if strict else v <= max))
    if min is not None:
        out.append(Check(f"{key} {'>' if strict else '>='} {min:g}", v, v > min if strict else v >= min))
    return out


@register_filter("min_retention")
def min_retention(m, min=0.3):
    """Guard against degenerate results: both labels keep at least `min` of their initial visible strands
    (FM or trim did not wipe out the highlight or the base)."""
    v = _get(m, "derived.min_label_retention")
    return [Check(f"min_label_retention >= {min:g}", v, v >= min)]


_SAFE_FUNCS = dict(min=min, max=max, abs=abs, sqrt=math.sqrt, log=math.log, isnan=math.isnan)
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)+")


@register_filter("expr")
def expr(m, rule):
    """A Python boolean expression over the flat metric names, e.g.
    "grid_score.mean_all < 0.08 and max(grid_score.per_black_area, grid_score.per_white_area) <= 0.3".
    Allowed functions: min, max, abs, sqrt, log, isnan."""
    names = {}

    def sub(mt):
        k = mt.group(0)
        if k not in m:
            raise KeyError(f"expr filter: unknown metric {k!r} (known: {', '.join(sorted(m)[:60])} ...)")
        var = f"_v{len(names)}"
        names[var] = m[k]
        return var

    code = _NAME.sub(sub, rule)
    ok = bool(eval(code, {"__builtins__": {}}, {**_SAFE_FUNCS, **names}))  # noqa: S307 (local research tool)
    return [Check(rule, float(ok), ok)]


# ---------------------------------------------------------------- spec parsing / evaluation

DEFAULT_SPEC = [dict(name="grid_score_abc", params=dict(a=0.1, b=0.25, c=0.1))]


def _num(s):
    s = s.strip()
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def parse_spec(spec):
    """None -> DEFAULT_SPEC; a list -> itself; a JSON file path / JSON string / short string -> list"""
    if spec is None:
        out = json.loads(json.dumps(DEFAULT_SPEC))
    elif isinstance(spec, list):
        out = spec
    elif os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            out = json.load(f)
    elif spec.strip().startswith(("[", "{")):
        out = json.loads(spec)
    else:
        out = []
        for part in spec.split("+"):
            name, _, rest = part.strip().partition(":")
            params = {}
            if name == "expr":
                params["rule"] = rest.partition("=")[2] if rest.startswith("rule=") else rest
            elif rest:
                for kv in rest.split(","):
                    k, _, v = kv.partition("=")
                    params[k.strip()] = _num(v)
            out.append(dict(name=name.strip(), params=params))
    if isinstance(out, dict):
        out = [out]
    for f in out:
        f.setdefault("params", {})
        if f["name"] not in FILTERS:
            raise KeyError(f"unknown filter {f['name']!r}; known: {sorted(FILTERS)}")
    return out


def spec_with_abc(spec, a=None, b=None, c=None):
    """override a/b/c of every grid_score_abc filter in the spec (the --a/--b/--c shortcuts)"""
    for f in spec:
        if f["name"] == "grid_score_abc":
            for k, v in (("a", a), ("b", b), ("c", c)):
                if v is not None:
                    f["params"][k] = v
    return spec


def spec_str(spec):
    return " + ".join(f["name"] + ("(" + ", ".join(f"{k}={v}" for k, v in f["params"].items()) + ")"
                                   if f["params"] else "") for f in spec)


def flatten_metrics(res):
    """result.json -> {"<metric>.<key>": number} plus derived keys:
        derived.highlight_retention / base_retention   visible strands of a label / its initial strands
        derived.min_label_retention                    min of the two
        derived.max_area_score / min_area_score        max / min of per_black_area and per_white_area"""
    m = {}
    for name, d in (res.get("metrics") or {}).items():
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                m[f"{name}.{k}"] = v
    if "strands.n" in m:
        n, f0 = m["strands.n"], m.get("strands.init_highlight_frac", 0.0)
        hi0, lo0 = f0 * n, (1 - f0) * n
        hr = m.get("strands.visible_highlight", 0) / hi0 if hi0 > 0 else 0.0
        br = m.get("strands.visible_base", 0) / lo0 if lo0 > 0 else 0.0
        m.update({"derived.highlight_retention": hr, "derived.base_retention": br,
                  "derived.min_label_retention": min(hr, br)})
    if "grid_score.per_black_area" in m:
        pb, pw = m["grid_score.per_black_area"], m["grid_score.per_white_area"]
        m.update({"derived.max_area_score": max(pb, pw), "derived.min_area_score": min(pb, pw)})
    return m


def evaluate(spec, m):
    """-> (passed, [check dicts with the filter name])"""
    checks = []
    for f in spec:
        for ch in FILTERS[f["name"]](m, **f["params"]):
            checks.append(dict(filter=f["name"], **ch.as_dict()))
    return all(c["ok"] for c in checks), checks
