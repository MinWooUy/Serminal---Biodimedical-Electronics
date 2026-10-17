"""Checks of scripts/final_eval_ds2.py WITHOUT evaluating DS2 (DS2 is locked; only the user runs it once).

Static (AST) checks of main(), unit tests of the pure bookkeeping functions on synthetic data, and a
smoke test of run() on ONE DS1 record.
"""

import ast
import importlib.util

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

PATH = REPO_ROOT / "scripts" / "final_eval_ds2.py"
SRC = PATH.read_text(encoding="utf-8")


def _module():
    spec = importlib.util.spec_from_file_location("final_eval_ds2", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # import only: no data access at import time (checked below)
    return mod


def _main_node():
    return next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")


# ---------- static checks of main()

def test_ds2_lock_is_the_first_statement_of_main():
    first = _main_node().body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", None) == "require_ds2_access"


def test_no_data_access_at_import_time():
    tree = ast.parse(SRC)
    for node in tree.body:                      # module level: imports, constants, defs, docstring, main guard
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Expr, ast.If, ast.Assign)), node
    names = {a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "load_record" not in names and "get_ds2_records" not in names


def test_script_never_sets_the_env_var():
    assert "environ[" not in SRC and "putenv" not in SRC and "setdefault" not in SRC


def test_main_checks_clean_tree_and_records_git_head_before_running():
    body = ast.unparse(_main_node())
    i_status, i_exit = body.find("git', 'status', '--porcelain'"), body.find("sys.exit(2)")
    i_rev, i_run = body.find("git', 'rev-parse', 'HEAD'"), body.find("run(get_ds2_records()")
    assert -1 not in (i_status, i_exit, i_rev, i_run)
    assert i_status < i_exit < i_run and i_rev < i_run
    assert "prefix='FINAL_DS2'" in body and "git_rev=git_rev" in body


def test_main_passes_only_ds2_records_after_the_lock():
    calls = [n for n in ast.walk(_main_node()) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run"]
    assert len(calls) == 1 and getattr(calls[0].args[0].func, "id", None) == "get_ds2_records"


# ---------- bookkeeping on synthetic data

def test_e2e_counts_q_and_vf_are_removed():
    mod = _module()
    NO = mod.NO_PRED
    ann_pos = np.array([100, 400, 700, 1000, 1300, 2500])
    ann_aami = np.array([0, -1, 2, 0, 1, 3])                 # N, Q, V, N (in VF), S (missed), F
    vf = np.array([[950, 1100]])
    det_pos = np.array([102, 402, 703, 1001, 1600, 1900, 2200, 2502])
    det_pred = np.array([0, 2, 2, 1, 1, 0, NO, NO])           # 402 matches Q; 1001 in VF; 2200 unclassified FP;
    c = mod.e2e_counts(ann_pos, ann_aami, vf, det_pos, det_pred, tol=54)   # 2502 matched F but unclassified
    assert c["y"].tolist() == [0, 2] and c["p"].tolist() == [0, 2]         # Q pair and VF pair removed
    assert c["fn"].tolist() == [0, 1, 0, 0]                  # S at 1300 missed; VF N and Q not counted
    assert sorted(c["fp_pred"].tolist()) == [0, 1]           # FP 1600 (S) and 1900 (N); unclassified FP dropped


def test_gross_metrics_definitions():
    mod = _module()
    y = np.array([0, 0, 1, 2, 3])
    p = np.array([0, 2, 1, 2, 2])                             # true F predicted V
    g = mod.gross_metrics(y, p, fp_pred=np.array([0, 1, 2]), fn=np.array([0, 1, 0, 0]))
    # N: TP 1, Se 1/2, +P 1/(1 + 1 FP predicted N) -> FP detections predicted N DO count for macro-F1
    assert g["per_class"]["N"]["ppv"] == 0.5 and g["per_class"]["N"]["se"] == 0.5
    # S: TP 1, Se 1/(1 + 1 missed), +P 1/(1 + 1 FP predicted S)
    assert g["ec57"]["SVEB"]["se"] == 0.5 and g["ec57"]["SVEB"]["ppv"] == 0.5
    # V: TP 1, predicted V = 3 (incl. true N and true F) + 1 FP; EC57 VEB +P excludes the F->V beat
    assert g["per_class"]["V"]["ppv"] == 1 / 4 and g["ec57"]["VEB"]["ppv"] == 1 / 3
    assert abs(g["macro_f1"] - np.mean([v["f1"] for v in g["per_class"].values()])) < 1e-12


# ---------- smoke test of run() on DS1

@pytest.mark.needs_data
def test_run_smoke_on_one_ds1_record():
    """Smoke test of the evaluation code path on a DS1 record (NOT DS2, NOT the final evaluation)."""
    import glob
    from pathlib import Path
    lite = sorted(glob.glob(str(REPO_ROOT / "results" / "*_final_A5_svdb_n25_lite_e3")))
    full = sorted(glob.glob(str(REPO_ROOT / "results" / "*_final_cnn_full")))
    if not lite or not full:
        pytest.skip("final models not trained")
    res = _module().run([208], Path(lite[-1]), Path(full[-1]), prefix="SMOKE_DS1_eval_code", git_rev="smoke")
    assert res["detector"]["150ms"]["tp"] > 2800
    for part in ("annotation", "end_to_end_matched", "end_to_end_gross"):
        assert set(res[part]) == {"full_float", "lite_float", "lite_integer"}
    g = res["end_to_end_gross"]["lite_integer"]
    assert 0 <= g["macro_f1"] <= 1 and g["per_class"]["V"]["se"] > 0.5


# ---------- provenance: frozen model files and weights.h

def test_main_verifies_provenance_before_running():
    body = ast.unparse(_main_node())
    i_ver, i_run = body.find("verify_provenance("), body.find("run(get_ds2_records()")
    assert -1 not in (i_ver, i_run) and i_ver < i_run and "provenance=provenance" in body


def test_run_uses_the_saved_int8_parameters_and_never_requantises():
    run = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    src = ast.unparse(run)
    assert "load_quant_model(fl / 'lite_int8.npz')" in src
    assert "quantize_model" not in src and "fold_model" not in src


def _final_dirs():
    import glob
    from pathlib import Path
    lite = sorted(glob.glob(str(REPO_ROOT / "results" / "*_final_A5_svdb_n25_lite_e3")))
    full = sorted(glob.glob(str(REPO_ROOT / "results" / "*_final_cnn_full")))
    if not lite or not full or not (REPO_ROOT / "export" / "final_models_manifest.json").exists():
        pytest.skip("final models / manifest not available")
    return Path(lite[-1]), Path(full[-1])


@pytest.mark.needs_data
def test_weights_h_is_generated_from_the_final_lite_int8_and_manifest_matches():
    fl, ff = _final_dirs()
    h = _module().verify_provenance(fl, ff)                 # exits (SystemExit) on any mismatch
    import hashlib
    assert h["weights_h_sha256"] == hashlib.sha256((REPO_ROOT / "export" / "weights.h").read_bytes()).hexdigest()
    assert "lite_int8.npz" in h["lite"] and "model.pt" in h["full"]


@pytest.mark.needs_data
def test_verify_provenance_refuses_tampered_models(tmp_path):
    import shutil
    fl, ff = _final_dirs()
    tl, tf = tmp_path / fl.name, tmp_path / ff.name
    shutil.copytree(fl, tl)
    shutil.copytree(ff, tf)
    mod = _module()
    mod.verify_provenance(tl, tf)                           # identical copies pass
    p = tl / "lite_int8.npz"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit) as e:
        mod.verify_provenance(tl, tf)
    assert e.value.code == 3
