from __future__ import annotations

PYHS3_NONCOMPILED = "pyhs3_noncompiled"
PYHS3_COMPILED = "pyhs3_compiled"
ROOFIT = "roofit"

ENGINE_LABELS = {
    PYHS3_NONCOMPILED: "pyHS3 non-compiled (PyTensor)",
    PYHS3_COMPILED: "pyHS3 compiled (JAX)",
    ROOFIT: "RooFit",
}

ENGINE_ORDER = (PYHS3_NONCOMPILED, PYHS3_COMPILED, ROOFIT)

FIXED_INPUT = "fixed"
VARYING_INPUT = "varying"
INPUT_MODES = (FIXED_INPUT, VARYING_INPUT)

SCALAR_PDF = "scalar_pdf"
POINTWISE_NLL = "pointwise_nll"
BATCHED_NLL = "batched_full_dataset_nll"
