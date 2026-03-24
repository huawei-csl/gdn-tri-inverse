# Fast matrix inversion on NPU with application to Gated DeltaNet 

**TL;DR** We created a matrix inversion kernel that is **3x faster** than the current Triton kernel in SGLang and vLLM-Ascend. The new kernel speeds up Gated DeltaNet by **20%~40% end-to-end** on NPU while maintaining (in some cases even improving) numerical accuracy.

- Date: 2026/03/25
- Team: Aleksandros Sobczyk, Gioele Gottardo, Filip Skogh, Mirko De Vita, Christos Konstantinos Matzoros, Anastasios Zouzias, Jiawei Zhuang

**To reproduce all results shown in this guide**, see:
- [Reproducible benchmark scripts](https://github.com/huawei-csl/gdn-tri-inverse)
- [Optimized C++ kernel in PTO-ISA](https://github.com/huawei-csl/pto-kernels/blob/v0.1.2/csrc/kernel/kernel_tri_inv_rec_unroll.cpp)
- [Educational Python kernel demo](https://github.com/huawei-csl/pto-dsl/tree/0.1.1/examples/aot/fast_inverse)

# Outline

- [Motivation and performance results](#motivation-and-performance-results)
- [Why do LLMs need matrix inverse? Brief recap of math and code](#why-do-llms-need-matrix-inverse-brief-recap-of-math-and-code)
- [Designing fast and accurate triangular inversion algorithms using matrix units](#designing-fast-and-accurate-triangular-inversion-algorithms-using-matrix-units)
  - [AI Accelerators and Ascend NPUs](#ai-accelerators-and-ascend-npus)
  - [Desired algorithm properties](#desired-algorithm-properties)
  - [A first attempt via backward substitution: Vectorized and Matrix-based Column-Sweep (VCS and MCS)](#a-first-attempt-via-backward-substitution-vectorized-and-matrix-based-column-sweep-vcs-and-mcs)
  - [A very fast, matrix product-based algorithm](#a-very-fast-matrix-product-based-algorithm)
  - [A more stable matrix-based algorithm: Revisiting Bunch and Hopcroft (MBH)](#a-more-stable-matrix-based-algorithm-revisiting-bunch-and-hopcroft-mbh)
  - [The best of both worlds: combining the speed of MCH and the stability of MBH](#the-best-of-both-worlds-combining-the-speed-of-mch-and-the-stability-of-mbh)
  - [Summary of methods](#summary-of-methods)
- [Deep-dive on Ascend 910B implementations](#deep-dive-on-ascend-910b-implementations)
  - [Low-level implementation of MXR using PTO-ISA](#low-level-implementation-of-mxr-using-pto-isa)
  - [Efficiently moving diagonal blocks between L1 and L0](#efficiently-moving-diagonal-blocks-between-l1-and-l0)
  - [Double-buffering and intra-core parallel/asynchronous execution](#double-buffering-and-intra-core-parallelasynchronous-execution)
- [End-to-end speed-up for chunkwise Gated DeltaNet](#end-to-end-speed-up-for-chunkwise-gated-deltanet)
- [Appendix](#appendix)
  - [Background on floating point and stability analysis](#background-on-floating-point-and-stability-analysis)
  - [Detailed stability analysis of the methods](#detailed-stability-analysis-of-the-methods)
- [Bibliography](#bibliography)

# Motivation and performance results

[Gated DeltaNet (GDN) architecture](https://arxiv.org/abs/2412.06464) is a popular choice for long-context LLMs, notably the [Qwen3.5 series](https://huggingface.co/collections/Qwen/qwen35) and [Kimi-Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct). GDN and many variants require inversion of a triangular matrix during their [chunkwise algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/#a-chunkwise-algorithm-for-deltanet) (for long-context prefill and training):

<p align="center">
  <img src="./fig/kda_table.png" alt="kda_table" style="width: 80%; max-width: 1000px;" />
</p>

(table from [Kimi Linear paper](https://arxiv.org/abs/2510.26692))

This triangular inversion takes **~40%** of the time on NPU, according to profiling of [optimized tilelang GDN](https://github.com/tile-ai/tilelang-ascend/tree/ede78f814e5e5dfcbfe783b79f988e6b6e375a86/examples/linear_attention_and_rnn#optimize-results):

<p align="center">
  <img src="./fig/tilelang_gdn_breakdown.png" alt="tilelang_gdn" style="width: 70%; max-width: 600px;" />
</p>

On our NPU, a larger chunk size like 128 improves the FLOP utilization of the matrix multiplication unit, effectively speeding up most GDN components except for the triangular inversion phase. While most components like `chunk_h` and `chunk_o` are largely tiled matmuls, triangular inversion relies on a forward-substitution algorithm that cannot be executed on matrix units.

To resolve this major bottleneck, we provide a **fast and numerically stable** triangular inverse kernel that is **3x faster** than the Triton implementation [in SGLang's Ascend backend](https://github.com/sgl-project/sgl-kernel-npu/tree/2026.03.01.post1/python/sgl_kernel_npu/sgl_kernel_npu/fla) and [in vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/v0.17.0rc1/vllm_ascend/ops/triton/fla) for chunk sizes up to 64 (the largest chunk size supported by the hard-coded Triton kernel).

<p align="center">
  <img src="./fig/vs_triton_chunk64.png" alt="vs_triton" style="width: 70%; max-width: 600px;" />
</p>

(The Y-axis "effective bandwidth" is inversely proportional to kernel time; the theoretical peak BW assumes only reading inputs and writing outputs while ignoring compute, in which case BW can approach HBM peak ~1 TB/s.)

It is also **3/3/1.5x faster** than the [optimized tilelang-ascend implementation](https://github.com/tile-ai/tilelang-ascend/tree/786a5ef0df8e98da97bcd51440ab55a8c8253e2c/examples/linear_attention_and_rnn/opt_gdn) for chunk sizes 32/64/128 respectively, and is more flexible w.r.t. shapes and data layouts (the tilelang kernel uses a fully static shape and an easier ["head-first" layout](https://github.com/fla-org/flash-linear-attention/pull/338), so it cannot be used in production yet).

<p align="center">
  <img src="./fig/vs_tilelang_chunk128.png" alt="vs_tilelang" style="width: 70%; max-width: 600px;" />
</p>

The new kernel also gives a substantial speedup for the entire GDN layer (extracted from SGLang code; other non-inverse parts still use the original Triton implementation):

<p align="center">
  <img src="./fig/gdn_breakdown.png" alt="e2e_gdn" style="width: 70%; max-width: 600px;" />
</p>

# Why do LLMs need matrix inverse? Brief recap of math and code

[GDN architecture](https://arxiv.org/abs/2412.06464) involves inversion of a lower-triangular matrix:

<p align="center">
  <img src="./fig/gdn_formula.png" alt="gdn_formula" style="width: 70%; max-width: 800px;" />
</p>

This inversion comes from [Accumulating Householder Transformations](https://dl.acm.org/doi/10.1145/1141885.1141886) (a good math refresher is Chapter 5.1, Householder Transformations, in the classic [Golub & van Loan book](https://epubs.siam.org/doi/book/10.1137/1.9781421407944)). Except for this inversion, other operations in the chunkwise algorithm are largely tiled matrix multiply-and-add operations, which are naturally mapped to matrix units such as the Cube unit on NPU and Tensor Cores on GPU.

Inside Hugging Face's [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py), the inversion is done by [forward substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_substitution) inside [torch_chunk_gated_delta_rule](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L368-L371), in the eager Torch fallback path (which runs on CPU/NPU/GPU):

```python
attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
```

We can confirm its behavior with a simple test:

```python
import torch
def solve_attn(attn, chunk_size=4):
    attn = attn.clone() # avoid in-place changes
    for i in range(1, chunk_size):
        row = attn[i, :i].clone()  # ignore broadcast dimensions here
        sub = attn[:i, :i].clone()
        attn[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn

torch.manual_seed(0)
c = 4   # can change to 8/16/32/...
A = torch.tril(torch.rand(c, c), diagonal=-1)
A_solve = solve_attn(A)
I = torch.eye(c)
print((I - A) @ (I + A_solve))  # equals the identity matrix
```

On GPU, fused Triton kernels are available: [solve_tril](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/utils/solve_tril.py) is called inside [chunk_gated_delta_rule_fwd](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/gated_delta_rule/chunk.py#L48).

On NPU, production frameworks like vLLM/SGLang use Triton-Ascend to compile [a similar kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/2026.03.01.post1/python/sgl_kernel_npu/sgl_kernel_npu/fla/solve_tril.py).


# Designing fast and accurate triangular inversion algorithms using matrix units

## AI Accelerators and Ascend NPUs

Modern hardware accelerators incorporate different types of compute units to satisfy the requirements of AI applications. Typically, they contain:
- Matrix multiplication units, which perform small matrix products $C=AB$ as a single instruction.
- SIMT/SIMD cores (vector-like processors) used for element-wise arithmetic/logic instructions, activation functions (ReLU, GeLU, etc.), and many other operations.
- Scalar cores, for performing basic logic and arithmetic operations on single elements.

Many commercially available architectures follow this paradigm, including the Ascend architecture. Our kernels are programmed in [PTO-ISA](https://gitcode.com/cann/pto-isa), which works for both current Ascend 910B and the upcoming [Ascend 950](https://gitcode.com/cann/community/tree/master/events/meetup/slides/950/20260316).
The experiments were executed on an Ascend 910B architecture, which consists of a main memory module and multiple *AI-cores* with local scratchpad memories, as illustrated in the following figure.
Each AI-core consists of two *AIV* (vector cores) and one *AIC* (Cube) core. Both AIV and AIC include scalar units for basic operations.
For more information on the Ascend architecture we refer to the [official documentation](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html).

<p align="center">
  <img src="./fig/ascend-architecture.png" alt="Ascend architecture diagram" style="width: 80%; max-width: 600px; background-color: rgb(255, 255, 255);" />
</p>

## Desired algorithm properties

Designing efficient algorithms on modern HW is a challenging task, which entails optimizing different aspects (simultaneously):
1) *Accuracy* and *numerical stability*: The algorithms need to return accurate solutions, and to be robust and reliable against numerical errors for all possible inputs.
2) *Speed* and *parallelism*: Algorithms need to be fast and to take advantage of the computational capabilities of the underlying hardware. 
3) *Memory*: Memory usage for temporary / auxiliary computations needs to be minimized (ideally none).

Hereafter we discuss several methods for inverting triangular matrices that lend themselves to efficient NPU implementations. For all algorithms we report:
- Their **complexity**, which is the number of basic operations they execute (number of matrix products, vector instructions, scalar ops, etc.),
- Their **numerical stability**, which quantifies how robust they are against numerical errors (see also the [Appendix](#background-on-floating-point-and-stability-analysis) for a refresher on numerical stability).

## A first attempt via backward substitution: Vectorized and Matrix-based Column-Sweep (VCS and MCS)

The column-sweep method is the standard method of performing forward substitution in a column-oriented manner (see e.g. ref. [4]). The method can be easily implemented in NumPy as shown below.

- **Complexity**: $n$ **vector ops** of length $n,n-1,n-2,...,2,1$  $\rightarrow O(n^2)$ flops
- **Stability**: The algorithm is **numerically stable**, as we defined previously. Details about the proof (and stricter error bounds) can be found in [4].

<details>
<summary>VCS NumPy code</summary>

```python
import numpy as np

def tri_inv_vcs(U: np.ndarray) -> np.ndarray:
    """
    Vector Column-Sweep algorithm which computes the 
    inverse of A = I + U, where: 
      I is the identity
      U is strictly upper triangular
    """
    n = U.shape[0]
    A_inv = np.zeros_like(U, dtype=U.dtype)

    for j in range(n):
        A_inv[j,j] = 1.0
    
        for k in range(n - 1, 0, -1):
            A_inv[:k, j] -= U[:k, k] * A_inv[k, j]

    return A_inv
```

</details>

The main problem of this method is that it does not take advantage of matrix products and the AIC cores.
However, it is known that it actually can be written in a matrix formulation (see Eqs. 3.8/3.9 in Section 3.2.1 of [4]). It is best explained by an example. 

Let A be a matrix of size 3 as below:

$$
A =
\begin{pmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
0 & 0 & 1
\end{pmatrix}.
$$

One can easily verify that the inverse of A can be written as

$$
A^{-1} = M_1 M_2 =
\begin{pmatrix}
1 & -2 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}\begin{pmatrix}
1 & 0 & -3 \\
0 & 1 & -4 \\
0 & 0 & 1
\end{pmatrix}.
$$

In general, the `MCS` algorithm below generates the matrices $M_{n-1}M_{n-2}\dots M_{1}$ and performs the chained matrix products. The main advantage of this approach compared to `VCS` is that `MCS` can take advantage of a matrix multiplication unit. One drawback is that, for matrix size $n$, it requires $n-1$ matrix products.

- **Complexity**: $n-1$ **matmuls** of size $n\times n$, $O(n)$ **vector ops** of length $O(n)$
- **Stability**: This algorithm is also **numerically stable**, like VCS. However, the constants might slightly differ, due to the differences in the numerical errors of the matmuls vs scalar operations.

<details>
<summary>MCS NumPy code</summary>

```python
import numpy as np

def tri_inv_mcs(U: np.ndarray) -> np.ndarray:
    """
    MCS Algorithm. Uses matrix products to compute the 
    inverse of A = I + U, where: 
      I is the identity
      U is strictly upper triangular
    """
    n = U.shape[0]
    I = np.eye(n, dtype=U.dtype)
    
    U = I - U
    A_inv = I.copy()
    
    for k in reversed(range(n)):
        M_k = I.copy()
        M_k[:, k] = U[:, k]
        A_inv = M_k @ A_inv
    
    return A_inv
```

</details>

## A very fast, matrix product-based algorithm
The main drawback of the MCS algorithm is that it executes $O(n)$ matrix products of size $n\times n$. 
The authors of [9] propose an alternative algorithm, taking advantage of the fact that the size of matrix A is always chosen as a power of 2 (usually 16, 32, or 64).

<p align="center">
  <img src="./fig/invtrick_formula1.png" alt="invtrick_formula1" width="100%" />
</p>
<p align="center">
  <img src="./fig/invtrick_formula2.png" alt="invtrick_formula2" width="100%" />
</p>

It comes from the [Cayley-Hamilton theorem](https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem), which gives a formulation to compute the inverse of a matrix with respect to its characteristic polynomial, i.e.:
$$
(I+U)^{-1}=I-U+U^2-U^3+\dots+(-1)^nU^n.
$$ 

Intuitively, this is analogous to [matrix power series](https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix#Power_series) together with [fast exponentiation by squaring](https://en.wikipedia.org/wiki/Exponentiation_by_squaring). We will refer to this algorithm as "MCH", from the acronym "Matrix Cayley-Hamilton".
In total, it requires $O(\log(n))$ matrix products:

**Algorithm MCH**$(L)$:
1. $I\gets$ identity matrix of size $n$
2. $X \gets I - L$
3. $Y \gets L$
4. for $i=1,...,\lfloor \log(n) \rfloor-1$:
    - $Y \gets Y\cdot Y$
    - $X \gets X + X \cdot Y$
5. return $X$

We can easily implement it and confirm its correctness in Python, using the default `float32` precision. 

<details>
<summary>MCH NumPy code</summary>

```python
import numpy as np
from numpy.linalg import inv

def is_power_of_2(c):
    return (c != 0) and (c & (c-1) == 0)

def strict_lower(c=4, seed=0):
    return np.tril(np.random.rand(c, c), k=-1)

def tri_inv_mch(A):
    """
    Compute (I + A)^{-1} without explicit inversion
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    c = A.shape[0]
    assert is_power_of_2(c) and c >= 4
    log2_c = int(np.log2(c))
    I = np.eye(c)
    X, Y = (I - A, A @ A)
    for i in range(log2_c - 1):
        X, Y = (X + X @ Y, Y @ Y)
    return X

for c in [4, 8, 16, 32, 64]:
    A = strict_lower(c)
    A_inv_ref = inv(A + np.eye(c))
    A_inv = tri_inv_mch(A)
    assert np.allclose(A_inv, A_inv_ref)  # all pass
```
</details>

This trick allows tile-based frameworks to compute `inv(I+A)` quickly, without fine-grained forward substitution that must run on scalar and vector units.


### NPU kernel implementation in PTO Python DSL

Due to the simplicity of this algorithm, we can straightforwardly implement it on NPU using PTO-DSL. See the full code [in this `fast_inverse/basic_dense` example](https://github.com/huawei-csl/pto-dsl/blob/0.1.1/examples/aot/fast_inverse/basic_dense/inverse_builder.py#L104-L158). For new readers, check out our previous [Matmul optimization guide](https://github.com/huawei-csl/pto-dsl/blob/0.1.1/examples/aot/matmul_optimization_guide/matmul_optim_guide.md) for a friendly introduction to Ascend NPU and PTO programming.

The key kernel code maps straight to the NumPy version:

```python
# Mirrors:
# for i in range(log2_c - 1):
#     X, Y = (X + X @ Y, Y @ Y)
for iter_idx in pto.range(c0, log2_blocksize, c1):
    tile.mov(x_l1, a_l0)
    tile.mov(i_l1, b_l0)
    tile.matmul(a_l0, b_l0, c_l0)

    tile.mov(y_l1, b_l0)
    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)  # x + x @ y

    with pto.if_context(iter_idx + c1 < log2_blocksize):
        tile.mov(c_l0, x_l1)
        tile.mov(y_l1, a_l0)
        tile.matmul(a_l0, b_l0, c_l0)
        tile.mov(c_l0, y_l1)  # y = y @ y
```

The only difference is that we explicitly require intermediate results to stay in the `L1` buffer, instead of going back to slow global memory. `tile.mov` clearly enforces this data locality. (For Triton-Ascend, this is trickier because `tl.load`/`tl.store` do not distinguish `L1` vs `L0`.)

This trivial implementation (without double-buffering and further optimization) already gives 3x speedup over the Triton baseline, measured by effective bandwidth utilization (Triton was 40~50 GB/s):

<p align="center">
  <img src="./fig/basic_inv_trick_bw.png" alt="basic_inv_trick_bw" width="50%" />
</p>


### Theoretical Analysis

- **Complexity**: $\approx 2\log_2(n)$ matrix products of size $n\times n$.
- **Stability**: This algorithm is **numerically unstable**. 

The effect of the instability is easier to see with a numerical example. In this figure, we test the maximum element-wise relative error of four different methods to compute the inverse of the matrix $I+L$, where $L$ is strictly lower-triangular with random uniform elements between $[0,1/2)$ (such values are realistic for GDN networks). The algorithms `np_inv`, `VCS`, and `MCS` return at least seven digits of accuracy in `float32`, and at least three digits of accuracy in `float16`, for all matrix sizes `16,32,64,128`. However, the numerical accuracy of MCH explodes very quickly for larger matrix sizes. For `n=128`, the max relative error of MCH is more than $10^3$ in `float32`, while for `float16` the returned solution contains `NaN` values.

<p align="center">
  <img src="./fig/mch-instability.png" alt="MCH instability" style="width: 80%; max-width: 800px;" />
</p>


The source of this instability is in fact the same one that makes the algorithm fast: the recursive-squaring part. To give more intuition, consider the following "bad example" where the errors can grow arbitrarily. Such an example is the all-ones lower-triangular matrix of size $n\times n$, whose inverse is also easy to derive analytically:
$$
L_{ones} = \begin{pmatrix}
1 & 0 & 0 & 0 & ... & 0 \\
1 & 1 & 0 & 0 & ... & 0 \\
1 & 1 & 1 & 0 & ... & 0 \\
\vdots & \vdots & \vdots & \ddots & 0 & 0 \\
1 & 1 & 1 & ... & 1 & 0 \\
1 & 1 & 1 & ... & 1 & 1 \\
\end{pmatrix}
,\qquad
L^{-1}_{ones} = \begin{pmatrix}
1 & 0 & 0 & 0 & ... & 0 \\
-1 & 1 & 0 & 0 & ... & 0 \\
0 & -1 & 1 & 0 & ... & 0 \\
\vdots & \ddots & \ddots & \ddots & 0 & 0 \\
0 & ... & 0 & -1 & 1 & 0 \\
0 & ... & 0 & 0 & -1 & 1 \\
\end{pmatrix}
$$
The condition number of the matrix $L_{ones}$ is very small. However, during the execution of the algorithm, already at the *fourth iteration*, the matrix $Y$ will contain entries that overflow to **inf** in **fp16** precision! Eventually, by using standard tools for numerical analysis, one can show that the floating point errors of this algorithm grow *exponentially* with respect to $n$ (this is left as an exercise for the interested readers).

## A more stable matrix-based algorithm: Revisiting Bunch and Hopcroft (MBH)

While the `MCH` algorithm is very fast and efficiently utilizes matrix products, its severe instability makes it unusable in practice. Thankfully, we can achieve high accuracy and good performance simultaneously by using an alternative algorithm.

A different way to use matrix products as basic operations to invert a triangular matrix is the following:
$$A^{-1} = \begin{pmatrix} A_{11} & A_{12} \\ 0 & A_{22}\end{pmatrix}^{-1} = \begin{pmatrix} A^{-1}_{11} & -A^{-1}_{11}A_{22}A_{22}^{-1} \\ 0 & A_{22}^{-1}\end{pmatrix}$$

At each step, it involves computing the inverses of two matrices of half the size, and two matrix products of half the size.

In order to take advantage of the matrix cores efficiently, the recursion needs to be "unrolled", in order to group together small matrix products that occur at the lowest levels of the recursion. The unrolled version is explained in the Appendix.

<details>
<summary>MBH NumPy code</summary>

```python
def even_blocks(A, bsz):
    n = A.shape[0]
    B = np.zeros((n, n), dtype=A.dtype)
    for idx in range(0, n, 2 * bsz):
        B[idx:idx + bsz, idx:idx + bsz] = A[idx:idx + bsz, idx:idx + bsz]
    return B

def odd_blocks(A, bsz):
    n = A.shape[0]
    B = np.zeros((n, n), dtype=A.dtype)
    for idx in range(bsz, n, 2 * bsz):
        B[idx:idx + bsz, idx:idx + bsz] = A[idx:idx + bsz, idx:idx + bsz]
    return B

def tri_inv_mbh(A, X = None, starting_block_size = 1):
    MA = -A
    n = A.shape[0]
    I = np.eye(n).astype(A.dtype)
    if X is None:
        X = I.copy()
    block_size = starting_block_size
    while block_size < n:
        LX = even_blocks(X, block_size)
        RX = odd_blocks(X, block_size)
        X = (LX @ MA + I) @ RX + LX
        block_size = block_size * 2
    return X
```

</details>

Analysis:
- **Complexity**: $\approx 2\log_2(n)$ matrix products of size $n\times n$.
- **Stability**: The algorithm is **logarithmically stable** (a proof can be found in [2]). 

## The best of both worlds: combining the speed of MCH and the stability of MBH

We finally describe a hybrid algorithm that takes advantage of both MBH and MCH. The idea is the following:

- Use MCH in order to compute the small inverses of the diagonal blocks of A, where each block has size 16 * 16. We choose this block size for two reasons:
    1. 16 * 16 is the fractal size of the cube unit for fp16
    2. If we choose a larger block size, the algorithm MCH is **very unstable**, and it can easily return NaNs even for well-conditioned matrices. This will be shown later in experiments.
- Use MBH with starting_block_size=16 to compute the final inverse.
Analysis:

- **Complexity**: $\approx 2\log_2(n)$ matrix products of size $n\times n$.
- **Stability**: Since the unstable part (MCH) is only executed for a **constant** number of iterations, the errors no longer grow exponentially with respect to $n$. Strictly speaking, the algorithm is **logarithmically stable**, just like MBH. However, the algorithm is generally expected to be more prone to errors than MBH, due to the initial unstable steps. 
<details>
<summary>MXR NumPy code</summary>

```python
def tri_inv_mxr(A):
    n = A.shape[0]
    block_size = 16
    DA = even_blocks(A, block_size) + odd_blocks(A, block_size)
    X = tri_inv_mch(DA, max_block_size=block_size)
    X = tri_inv_mbh(A, X, starting_block_size=block_size)
    return X
```

</details>

A clean implementation of the MXR algorithm in PTO Python DSL is available in 
[fast_inverse/block_inversion](https://github.com/huawei-csl/pto-dsl/tree/0.1.1/examples/aot/fast_inverse/block_inversion) (For educational purposes, without performance tuning, and recursing only one level. See the later section for the fully optimized version)


## Summary of methods

Below, we provide a summary of the methods discussed previously in terms of the number of matrix multiplications used and numerical stability.
| **Method** | **Description** | **# MatMuls** | **Stability** | **Notes** |
|----|----|----|----|----|
| **VCS** | Vector column-sweep (forward substitution) | **0** | stable [6] | Uses only scalar/vector operations (axpy-style) |
| **MCS** | Matrix-based column-sweep | $n-1$ | stable [6] | One MM per output column |
| **MCH** | Cayley–Hamilton / inverse trick | **$\approx 2\log n $** | unstable | Very fast and simple, but unstable |
| **MBH** | Unrolled recursion (fast triangular inversion) | **$\approx 2\log n$** | log-stable [2] | Needs structured DataCopies for efficient implementation |
| **MXR** | Mixed MCH+MBH | $\approx 2\log n$  | log-stable [2] | Combines stability of MBH and speed of MCH. |

# Deep-dive on Ascend 910B implementations

In this section we provide details on low-level NPU implementations using PTO-ISA as the programming framework.

## Low-level implementation of MXR using PTO-ISA

Recall that the `MXR` algorithm has two parts:
1. The first part implements `MCH` to invert the small $(16\times 16)$ diagonal blocks of the matrix.
2. The second part uses the unrolled `MBH` algorithm to assemble the entire inverse.

For example, if every matrix multiplication instruction computes the product of two $16\times 16$ matrices, it is a severe waste to use such an instruction to repeatedly compute just the product of $2\times 2$ matrices. Ideally, we would like to group $8$ such $2\times 2$ products in a bigger $16\times 16$ product, by placing them in the diagonal blocks of the bigger matrices. 

To that end, we study an unrolled version of this algorithm, where we first copy the diagonal blocks of the matrix in two new matrices. Before diving into the details, we define some notation.

- Assuming some block_size$\in\{2, 4, 8, …, n/2\}$, we define the matrix
    $$
    D=\begin{pmatrix} X_{0,0} & 0 & \ldots &0 & 0 \\0 & X_{1,1} &\ldots & 0 & 0 \\0 & 0 & \ddots & 0 & 0 \\0 & 0 & \ldots & X_{b-2,b-2} & 0 \\ 0 & 0 & \ldots & 0 & X_{b-1,b-1}\end{pmatrix},
    $$ 
    that contains only the diagonal blocks of the matrix $X$. Each block $X_{i,i}$ has size $\text{block\_size} \times \text{block\_size}$, and there are $b=n/\text{block\_size}$ blocks in total.
    
- We now define two other block-diagonal matrices:
    - $L_X$ : Contains the “even” diagonal blocks: $[X_{0,0}, 0, X_{2,2}, 0, \ldots]$
    - $R_X$ : Contains the “odd” diagonal blocks: $[0, X_{1,1}, 0, X_{3,3}, 0, \ldots]$
    - Clearly, $L_X+R_X=D$.
A high-level description of the algorithm follows. We refer to it as "MBH", since one of the first appearances in the literature is by Bunch and Hopcroft [1].

**Algorithm Unrolled-MBH $(U)$:**
1. Initialize $X=I_{n\times n}$ and block_size=1
2. While block_size < n:
    1. Create $L_X,R_X$  as defined above, using structured DataCopy for efficiency.
    2. Update the matrix $X\gets  L_X + R_X-L_X \cdot U\cdot  R_X$
    3. block_size ← 2 * block_size
3. Return $X$.


The reason why `MCH` is used on $16\times 16$ blocks is two-fold:
- The `AIC` (Cube) cores of `Ascend` partition the input matrices into fractals of size $16\times 16$ for `fp16` data types. These fractals are the **smallest** unit that we can manipulate with standard data-copy / data-load instructions in `PTO-ISA`. Specifically, `TMOV` and `TEXTRACT` instructions.
- Due to the exponentially-increasing numerical instability, $16\times 16$ seems to be the largest "block-size" that we can efficiently invert using `MCH` while still retaining acceptable numerical behaviour (not optimal, but decently close to the machine precision).

## Efficiently moving diagonal blocks between L1 and L0

For step `1.` we need to implement an efficient method that copies the diagonal fractals of the input matrix from the `L1` memory to `L0A/L0B` memories. This can be done as follows (for simplicity, we hard-code the input type to `float16` and the fractal size to $16\times 16$, and we only show the case of `L0A`-"left tile"):
```cpp
/*
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each.
 * The src matrix lies in L1, while the dst matrix lies in L0A.
 * This kernel copies only the diagonal blocks (fractals) of size 16 * 16.
 */
template <uint32_t MatrixSize, typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyDiagonalFractalsL1ToL0(SrcL1TileT src, DstL0TileT dst) {
  constexpr uint32_t NumFractals = MatrixSize / 16;
  using FractalTileT = TileLeft<half, 16, 16>;
  FractalTileT fractals[NumFractals];
  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < NumFractals; ++i) {
    TASSIGN(fractals[i], starting_address + i * 16 * (MatrixSize + 16) * sizeof(half));
    TEXTRACT(fractals[i], src, i * 16, i * 16);
  }
}
```
Using this method, we can extract the $16\times 16$ diagonal blocks of a $64\times 64$ matrix in the following form:
$$
A = \begin{pmatrix}
A_{00} & A_{01} & A_{02} & A_{03} \\
0_{16\times 16} & A_{11} & A_{12} & A_{13} \\
0_{16\times 16} & 0_{16\times 16} & A_{22} & A_{23} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & A_{33} \\
\end{pmatrix}
\rightarrow
\text{CopyDiagonalFractalsL1ToL0}
\rightarrow
\begin{pmatrix}
A_{00} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & A_{11} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & A_{12} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & A_{13} \\
\end{pmatrix}
$$

Each of the blocks $A_{00},A_{11},A_{22},A_{33}$ is completely independent, so we can invert them in parallel using the `MCH` algorithm and exploiting matrix products of size $64\times 64$.

For step `2.`, we need a more involved method that copies either the `odd-indexed` or the `even-indexed` diagonal blocks of a matrix, where the block size varies between $16\times 16$ and $64\times 64$. We call this method `CopyOddOrEvenDiagonalBlocksL1ToL0`. For example, this method can copy the `odd` diagonal blocks of size $16\times 16$ of a $64\times 64$ matrix from `L1` to `L0` as follows:
$$
A = \begin{pmatrix}
A_{00} & A_{01} & A_{02} & A_{03} \\
0_{16\times 16} & A_{11} & A_{12} & A_{13} \\
0_{16\times 16} & 0_{16\times 16} & A_{22} & A_{23} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & A_{33} \\
\end{pmatrix}
\rightarrow
\text{odd blocks}
\rightarrow
\begin{pmatrix}
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & A_{11} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & A_{33} \\
\end{pmatrix},
$$
or the `even` blocks:
$$
A = \begin{pmatrix}
A_{00} & A_{01} & A_{02} & A_{03} \\
0_{16\times 16} & A_{11} & A_{12} & A_{13} \\
0_{16\times 16} & 0_{16\times 16} & A_{22} & A_{23} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & A_{33} \\
\end{pmatrix}
\rightarrow
\text{even blocks}
\rightarrow
\begin{pmatrix}
A_{00} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & A_{22} & 0_{16\times 16} \\
0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} & 0_{16\times 16} \\
\end{pmatrix}.
$$

## Double-buffering and intra-core parallel/asynchronous execution

In order to increase performance, we need to take advantage of asynchronous execution of data movements (`TMOVE` / `TEXTRACT`) between the L1 and L0 cache levels and compute instructions (`TMATMUL`) within the same `AIC` core. In the following figure we describe an example of how a naive serial execution of the `MCH` iteration can be executed efficiently, by having two buffers for `L0A`, `L0B`, and `L0C` matrices, and by overlapping computation and data-movements. For example, recall that the first step of the `MCH` iteration is the computation of $Y^2$. In PTO-ISA, it can be executed with the following instructions:
```cpp
TMOV(L0A_tile, Y_L1_tile);
TMOV(L0B_tile, Y_L1_tile);
TMATMUL(L0C_tile, L0A_tile, L0B_tile);
TMOV(Y_L1_tile, L0C_tile); // Now Y_L1_tile contains Y^2
```
By carefully identifying the dependencies between operations/memory accesses, and by using two buffers for `L0` matrices instead of one, we can significantly reduce the final runtime. The corresponding `MBH` part of `MXR` can also be executed efficiently with double-buffering. Note that, in the figure, the "depth" of the computation is reduced by half. Of course, this does not necessarily translate to `2x` speed-up, since different instructions require different numbers of cycles, but it significantly increases performance in practice.

<p align="center">
  <img src="./fig/inv-trick-tree.png" alt="inv-trick-tree" style="width: 80%; max-width: 800px;" />
</p>


# End-to-end speed-up for chunkwise Gated DeltaNet

Last but not least, we measure the performance of chunkwise Gated DeltaNet. We use as a baseline the official Triton kernels for SGLang [here](https://github.com/sgl-project/sgl-kernel-npu/blob/2026.03.01.post1/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py#L215), and we replace the triangular inverse with our fastest algorithm. One main challenge here is that the Triton kernel uses a different data layout for the inputs that correspond to the "seq-first" layout (`[batch, seq, head, hidden]`, marked as "BSND" below). This means that to integrate our kernel into SGLang we need two additional transpose operations. With the additional transposes we only achieved end-to-end `1.08x` speedup compared to Triton.

To avoid the transposition overhead we rewrote our kernel so that it can natively read the BSND layout. This was achieved by changing the strides in the memory accesses in PTO-ISA so that the reads are redirected to the correct address. This leads to a 1.18x speedup end-to-end.

<p align="center">
  <img src="./fig/GDN-layer-e2e.png" alt="GDN-layer-e2e" style="width: 60%; max-width: 600px;" />
</p>

Profiling with [torch-npu](https://gitcode.com/Ascend/pytorch) shows that the chain of small Triton kernels is bounded by kernel launch overhead in PyTorch eager mode, and the actual kernel execution takes little time. The host overhead can be much reduced by recording multiple kernels with [aclgraph](https://gitcode.com/Ascend/torchair/). If we consider only kernel execution time, swapping in our best inverse kernel speeds up the GDN layer by 1.4x: 

<p align="center">
  <img src="./fig/gdn_breakdown.png" alt="gdn-breakdown" style="width: 80%; max-width: 600px;" />
</p>


# Appendix

## Background on floating point and stability analysis
Floating point error analysis becomes more and more important in AI computing, especially with the recent trend of reducing the bits of precision in floating point formats. For the purposes of this note, we recall some basic definitions, and refer to the classic textbook of Higham [7] for further details.

In the standard floating point model, the floating point representation $fl(x)$ of a real number $x$ is a $(1+p+t)$-bit number:
$$fl(x) = \pm 2^e\times \left(\frac{m_1}{2} + \frac{m_2}{2^2}+...+\frac{m_t}{2^t}\right),$$
where:
- one bit stores the sign of the number $\pm$,
- $p$ bits are used to store the (integral) exponent $e$ in the so-called "biased" format,
- $t$ bits $m_1,m_2,...,m_t$ are used for the *significand* (also known as the *mantissa*).

For all *normalized* numbers it holds that: 
$$
fl(x)=(1+\theta)x,
$$ 
where 
$$
|\theta|\leq 2^{-t}.
$$
Recall that, roughly speaking, normalized numbers are all numbers that can be represented within the given exponent range.


Given two (normalized) floating point numbers $a,b$, floating point operations $\circ\in\{+,-,\times,/\}$ and square roots are assumed to satisfy:
$$ 
fl(a \circ b) = (1+\theta)(a\circ b),
$$
and
$$
fl\left(\sqrt{a}\right)=(1+\theta)\sqrt{a},
$$
where, again, $|\theta|\leq 2^{-t}$.

It is known that different numerical algorithms satisfy different *stability* properties, i.e., some algorithms are more prone to errors than others, even if they theoretically solve the exact same problem. A classic example is the Gram-Schmidt orthogonalization procedure for computing an orthonormal basis, which is known to be very unstable unless properly modified.

For the purposes of this article, we use the following notions of stability specifically for computing the **inverse of a triangular matrix** (more details can be found in [7] and [2], and references therein). For what follows, $c_1$ and $c_2$ are global constants (independent of the matrix size $n$), and $\kappa(A)=\|A\|\|A^{-1}\|\geq 1$ is the 2-norm condition number of $A$. 

- Given a matrix $A$ of size $n$, the goal is to approximate its inverse with a matrix $\widetilde A^{-1}$.
- **Numerically stable** inversion algorithms return solutions that satisfy:
$$
\| A^{-1} - \widetilde A^{-1}\| \leq c_1 n^{c_2} \cdot 2^{-t} \cdot \kappa(A)\cdot\|A^{-1}\| 
$$
- **Logarithmically stable** algorithms, on the other hand, satisfy:
$$
\| A^{-1} - \widetilde A^{-1}\| \leq c_1 n^{c_2} \cdot 2^{-t} \cdot \kappa^{P(\log(n))}(A)\cdot\|A^{-1}\|, 
$$
where $P(x)$ is a low-degree polynomial.
- **Unstable** algorithms do not satisfy any of these bounds (for example, errors can grow exponentially in $n$).

Evidently, logarithmically-stable algorithms are more prone to errors than numerically-stable ones.


## Detailed stability analysis of the methods

Here we give some more details regarding the algorithms' numerical stability for different precisions. We focus on `float16` and `float32` data types (currently supported by Ascend 910B). 

<!-- The mathematical analysis of the “forward” and “backward” errors that are guaranteed by each method can be found in the corresponding bibliography:

- **VCS** and **MCS** satisfy strong error guarantees, i.e., they are both **forward** and **backward stable.** The interested reader is referred to Reference [6, Section 2] for more details and explicit error bounds.
- The error bounds of **MBH** and **MXR** are dominated by the errors of the recursive triangular inversion algorithm. This algorithm satisfies a slightly weaker notion of stability than VCS and MCS, called *logarithmic stability,* because the final errors increase proportionally to the condition number of the matrix raised to a power of $O(\log n)$. We refer to [2, Section 3.1].
- The algorithm **MCH** is unstable. Even for well-conditioned matrices, the algorithm can return `NaN` due to the rapid growth of the elements of the intermediate matrices. This algorithm is only suitable for small matrices, e.g. up to size 16 * 16. -->

In our setup (LLM applications), the matrix input sizes of interest are `16, 32, 64` and `128`. We consider random strictly triangular matrices, where the elements are drawn independently and uniformly in the range `[0,1/2)`.

```python
A = 0.5 * np.tril(np.random.rand(n, n), k=-1)
```

In the following plots we report three different types of errors for `float16` and `float32` data types. We denote by $A^{-1}$  the true inverse of the matrix $A=I+L$, and by $\widetilde A^{-1}$ the inverse returned by each method.
- **Max element-wise absolute error**. This is mathematically equivalent to:
$$
\max_{i,j} |A^{-1}_{i,j} - \widetilde A^{-1}_{i,j}|.
$$
- **Max element-wise relative error**. This is defined as:
$$
\max_{i,j<i} \frac{|A^{-1}_{i,j} - \widetilde A^{-1}_{i,j}|}{|A^{-1}_{i,j}|}.
$$
- **Frobenius-norm relative error**. This can be seen as an "amortized" error (i.e., RMSE) over all $i,j$ and it gives an insight on the average number of correct digits per element. It is defined as follows:
$$
\frac{\|A^{-1} - \widetilde A^{-1}\|_F}{\|A^{-1}\|_F}.
$$

**Remark:** Note that individual examination of these metrics can be misleading. Collectively, they can give a better intuition on the numerical behaviour of an algorithm. For instance, if someone looks only at the max element-wise relative error, and it is large, then it would be easy to argue that the algorithm is not stable. However, if the corresponding Frobenius-norm relative error is small, it means that only a few "outliers" have a large relative error, while most of the other elements are well-approximated. This can be acceptable for many applications. At the same time, if the element-wise absolute error is very small, this means that only the very small values are heavily perturbed, while the larger elements ("heavy-hitters") are well approximated within acceptable error tolerances.


<p align="center">
  <img src="./fig/errors-fp16.png" alt="total-errors" style="width: 80%; max-width: 800px;" />
</p>

`MCH` is the only method that returns inaccurate solutions for sizes larger than `32`. The `MXR` algorithm achieves almost the same accuracy as the other, more stable methods, but at the same time it achieves the same computational efficiency as `MCH`: it requires only $\approx 2\log(n)$ matrix products while maintaining high accuracy.

# Bibliography

**[1]** Bunch, James R., and John E. Hopcroft. "Triangular factorization and inversion by fast matrix multiplication." *Mathematics of Computation* 28.125 (1974): 231-236.

**[2]** Demmel, James, Ioana Dumitriu, and Olga Holtz. "Fast linear algebra is stable." *Numerische Mathematik* 108.1 (2007): 59-91.

**[3]** Yang, Songlin, Jan Kautz, and Ali Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." *The Thirteenth International Conference on Learning Representations*.

**[4]** Gallopoulos, Efstratios, Bernard Philippe, and Ahmed H. Sameh. *Parallelism in matrix computations*. Dordrecht: Springer, 2016.

**[5]** Zhang, Yu, et al. *“KIMI LINEAR: An Expressive, Efficient Attention Architecture.”* [**arXiv preprint arXiv:2510.26692**](https://arxiv.org/pdf/2510.26692), 2025.

**[6]** Higham, Nicholas J. "The accuracy of solutions to triangular systems." *SIAM Journal on Numerical Analysis* 26.5 (1989): 1252-1265.

**[7]** Higham, Nicholas J. *Accuracy and stability of numerical algorithms*. SIAM, 2002.

**[8]** Kahan, William, and Joseph D. Darcy. "How Java’s floating-point hurts everyone everywhere." *ACM 1998 workshop on java for high-performance network computing*. Stanford University, 1998.

**[9]** Zhong, S., Xu, M., Ao, T. and Shi, G., 2025. [Understanding Transformer from the Perspective of Associative Memory](https://arxiv.org/abs/2505.19488). arXiv preprint arXiv:2505.19488.
