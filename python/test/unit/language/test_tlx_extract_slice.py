import pytest
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.compiler.errors import CompilationError


@triton.jit
def _extract_slice_without_layout(X, Y):
    offs = tl.arange(0, 32)
    x = tl.load(X + offs[:, None] * 32 + offs[None, :])
    y = tlx.extract_slice(x, [16, 32], [16, 0])
    rows = tl.arange(0, 16)
    tl.store(Y + rows[:, None] * 32 + offs[None, :], y)


def test_extract_slice_without_distributed_layout_reports_error():
    source = ASTSource(
        fn=_extract_slice_without_layout,
        signature={"X": "*fp16", "Y": "*fp16"},
    )

    with pytest.raises(CompilationError, match="requires a distributed source layout"):
        triton_compile(source, target=GPUTarget("hip", "gfx950", 32))
