[SGLang](https://github.com/sgl-project/sglang) is a high-performance framework for serving and benchmarking large language models (LLMs). It is designed to support end-to-end inference workflows, from model loading and request scheduling to optimized attention kernels and distributed execution.

We use SGLang as our reference and baseline for running the [Qwen3-Next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) model on Ascend NPUs. All the optimized kernels used by SGLang for Ascend, including the inverse, come from [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu).

The e2e evaluation for both performance and accuracy of the triangular inverse comes down to replacing one single [line](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py#L215) from sgl-kernel-npu and then using SGLang's infrastructure to perform all the tests.

# Running SGLANG on A2

In this section we present a setup that has been shown to be working on the .12 server for running SGLang.

## Running the sglang server

To run the SGLang server we use the SGLang docker image with CANN version 8.3 (the newer version with CANN 8.5 doesn't work, see...)

```bash
#!/bin/bash
#
SLANG_DOCKER_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann8.3.rc1-910b"

drun() {

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/ "$@"
}

drun --env "HF_ENDPOINT=https://hf-mirror.com" \
    ${SLANG_DOCKER_IMAGE} /bin/bash
```

In the docker image
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/latest/bisheng_toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --attention-backend ascend \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size 4 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 2048
```
After 5-10 minutes (if the model was already downloaded), you will see:
![image](uploads/4302ddb7cd0cd29111f4265902d3eaf6/image.png)

Once the server is up and running you can send requests from anywhere in the server with:
```python
import requests

port = 30000
url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "qwen/qwen3-next-80b-a3b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print(response.json())
```
The result will most likely be gibberish due to the poor accuracy of the inverse in sgl-kernel-npu.

## Model's accuracy

The accuracy of the inverse algorithm is crucial for running the Qwen3-Next model. Therefore we can study how different kernels implementing the inverse impact the e2e accuracy using the MMLU benchmark

To check and compare the model's accuracy for different implementations of the triangular inverse, clone [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) and [sglang](https://github.com/sgl-project/sglang).

Then run docker:
```bash

drun --env "HF_ENDPOINT=https://hf-mirror.com"  --volume ~/sgl-kernel-npu/:/tmp/sgl-kernel-npu/ --volume ~/sglang/:/tmp/sglang/   ${SLANG_DOCKER_IMAGE} /usr/bin/bash
```

From the docker image run the server:
```bash
python3 -m sglang.launch_server     --model-path Qwen/Qwen3-Next-80B-A3B-Instruct     --disable-cuda-graph --mem-fraction-static=0.7   --attention-backend ascend  --tp-size=8 --max-total-tokens 2048 --disable-radix-cache
```

From another window open docker and run the MMLU benchmark:
```
cd /tmp/sglang/benchmark/mmlu/
bash download_data.sh
python3 bench_sglang.py --nsub 10
```
This will run the benchmark on the model using the native implementation of GDN (very slow).
```
subject: abstract_algebra, #q:100, acc: 0.680
subject: anatomy, #q:135, acc: 0.793
subject: astronomy, #q:152, acc: 0.941
subject: business_ethics, #q:100, acc: 0.780
subject: clinical_knowledge, #q:265, acc: 0.838
subject: college_biology, #q:144, acc: 0.958
subject: college_chemistry, #q:100, acc: 0.630
subject: college_computer_science, #q:100, acc: 0.830
subject: college_mathematics, #q:100, acc: 0.670
subject: college_medicine, #q:173, acc: 0.769
Total latency: 2016.666
Average accuracy: 0.805
```

### Triton backend

In the lates version of `sgl-kernel-npu` the default version of GDN is the optimized triton-ascend kernel, therefore to run the triton backend we just need to install the latest version of `sgl-kernel-npu` in the docker image:

```bash
cd sgl-kernel-npu
bash build.sh -a kernels
pip install --force-reinstall output/sgl_kernel_npu*.whl
```
And run the sglang benchmark as before.
```
subject: abstract_algebra, #q:100, acc: 0.800
subject: anatomy, #q:135, acc: 0.807
subject: astronomy, #q:152, acc: 0.947
subject: business_ethics, #q:100, acc: 0.820
subject: clinical_knowledge, #q:265, acc: 0.891
subject: college_biology, #q:144, acc: 0.972
subject: college_chemistry, #q:100, acc: 0.670
subject: college_computer_science, #q:100, acc: 0.860
subject: college_mathematics, #q:100, acc: 0.770
subject: college_medicine, #q:173, acc: 0.861
Total latency: 436.129
Average accuracy: 0.855
```

### Fast inverse backend (vector colums sweep)
To run the fast inverse backend some modifications of `sgl-kernel-npu` are required. More precisely replace `solve_tril` with `fast_inv_tril_wrapper` in https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py#L215, where `fast_inv_tril_wrapper` is defined as follows:
```python
def fast_inv_tril_wrapper(
    A: torch.Tensor, cu_seqlens: Optional[torch.LongTensor] = None
):
    dtype = A.dtype
    B, T, H, BT = A.shape
    chunk_size = BT

    if cu_seqlens is not None:

        def get_pad_size(T, chunk_size):
            return (chunk_size - T % chunk_size) % chunk_size

        lengths = prepare_lens(cu_seqlens)

        chunks_per_seq = triton.cdiv(lengths, chunk_size)
        starts = (
            torch.cat([cu_seqlens.new_tensor([0]), chunks_per_seq.cumsum(0)[:-1]])
            * chunk_size
        )

        A_list = []
        for i in range(len(cu_seqlens) - 1):
            pad_size = get_pad_size(cu_seqlens[i + 1] - cu_seqlens[i], chunk_size)
            A_chunk = F.pad(
                A[:, cu_seqlens[i] : cu_seqlens[i + 1], :, :],
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    pad_size,
                    0,
                    0,
                ),
            )
            A_list.append(A_chunk)

        A = torch.cat(A_list, dim=1)
        A = A.transpose(1, 2).contiguous()
        padded_shape = A.shape
        A = A.view(-1, BT, BT)

        torch.npu.synchronize()
        A_inv = fast_inv_tril(-A)
        torch.npu.synchronize()

        A_inv = A_inv.view(padded_shape).transpose(1, 2).contiguous()
        A_inv_list = [
            A_inv[:, starts[i] : starts[i] + lengths[i], :, :]
            for i in range(len(cu_seqlens) - 1)
        ]
        A_inv = torch.cat(A_inv_list, dim=1).to(dtype)

        return A_inv
    else:
        padding_size = (chunk_size - T % chunk_size) % chunk_size
        A = F.pad(A, (0, 0, 0, 0, 0, padding_size, 0, 0))

        A = A.transpose(1, 2).contiguous()
        A = A.view(-1, BT, BT)

        torch.npu.synchronize()
        A_inv = fast_inv_tril(-A)
        torch.npu.synchronize()

        A_inv = (
            A_inv.view(B, H, -1, BT)[:, :, :T, :]
            .contiguous()
            .transpose(1, 2)
            .contiguous()
        )
        return A_inv.to(dtype)
```

Then reinstall `sgl-kernel-npu` again with:
```
bash build.sh -a kernels
pip install --force-reinstall output/sgl_kernel_npu*.whl
```
And run the accuracy benchmark again as before.
```
subject: abstract_algebra, #q:100, acc: 0.800
subject: anatomy, #q:135, acc: 0.807
subject: astronomy, #q:152, acc: 0.941
subject: business_ethics, #q:100, acc: 0.810
subject: clinical_knowledge, #q:265, acc: 0.894
subject: college_biology, #q:144, acc: 0.972
subject: college_chemistry, #q:100, acc: 0.670
subject: college_computer_science, #q:100, acc: 0.840
subject: college_mathematics, #q:100, acc: 0.780
subject: college_medicine, #q:173, acc: 0.855
Total latency: 491.175
Average accuracy: 0.852
```

### PTO Rec-unroll inverse

We will now run the Qwen3-Next model with our fastest inverse algorithm. To do so clone the [pto-kernels](https://gitlab.huaweirc.ch/zrc-von-neumann-lab/tcuscan/pto-kernels) repo and build the wheel:
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip3 install pyyaml setuptools pytest packaging
pip3 install -r requirements.txt
make build_wheel
```
Then install the wheel in docker. Remember that the python versions of the wheel and docker have to be the same. Finally run the e2e benchmark for the pto kernel with the following wrapper:
```python
from pto_kernels import pto_tri_inv_rec_unroll

def pto_rec_unroll_wrapper(
    A: torch.Tensor, cu_seqlens: Optional[torch.LongTensor] = None
):
    dtype = A.dtype
    B, T, H, BT = A.shape
    chunk_size = BT

    if cu_seqlens is not None:

        def get_pad_size(T, chunk_size):
            return (chunk_size - T % chunk_size) % chunk_size

        lengths = prepare_lens(cu_seqlens)

        chunks_per_seq = triton.cdiv(lengths, chunk_size)
        starts = (
            torch.cat([cu_seqlens.new_tensor([0]), chunks_per_seq.cumsum(0)[:-1]])
            * chunk_size
        )

        A_list = []
        for i in range(len(cu_seqlens) - 1):
            pad_size = get_pad_size(cu_seqlens[i + 1] - cu_seqlens[i], chunk_size)
            A_chunk = F.pad(
                A[:, cu_seqlens[i] : cu_seqlens[i + 1], :, :],
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    pad_size,
                    0,
                    0,
                ),
            )
            A_list.append(A_chunk)

        A = torch.cat(A_list, dim=1)
        A = A.transpose(1, 2).contiguous()
        padded_shape = A.shape
        A = A.view(-1, BT, BT).transpose(1, 2).contiguous()

        torch.npu.synchronize()
        A_inv = pto_tri_inv_rec_unroll(A.to(torch.float16)).to(torch.float32)
        torch.npu.synchronize()

        A_inv = A_inv.transpose(1, 2).contiguous().view(padded_shape).transpose(1, 2).contiguous()
        A_inv_list = [
            A_inv[:, starts[i] : starts[i] + lengths[i], :, :]
            for i in range(len(cu_seqlens) - 1)
        ]
        A_inv = torch.cat(A_inv_list, dim=1).to(dtype)

        return A_inv
    else:
        padding_size = (chunk_size - T % chunk_size) % chunk_size
        A = F.pad(A, (0, 0, 0, 0, 0, padding_size, 0, 0))

        A = A.transpose(1, 2).contiguous()
        A = A.view(-1, BT, BT).transpose(1, 2).contiguous()

        torch.npu.synchronize()
        A_inv = A_inv = pto_tri_inv_rec_unroll(A.to(torch.float16)).to(torch.float32)
        torch.npu.synchronize()

        A_inv = (
            A_inv.transpose(1, 2).contiguous().view(B, H, -1, BT)[:, :, :T, :]
            .contiguous()
            .transpose(1, 2)
            .contiguous()
        )
        return A_inv.to(dtype)
```
And run the accuracy benchmark again as before.
```
subject: abstract_algebra, #q:100, acc: 0.800
subject: anatomy, #q:135, acc: 0.807
subject: astronomy, #q:152, acc: 0.941
subject: business_ethics, #q:100, acc: 0.810
subject: clinical_knowledge, #q:265, acc: 0.887
subject: college_biology, #q:144, acc: 0.972
subject: college_chemistry, #q:100, acc: 0.670
subject: college_computer_science, #q:100, acc: 0.850
subject: college_mathematics, #q:100, acc: 0.780
subject: college_medicine, #q:173, acc: 0.861
Total latency: 453.995
Average accuracy: 0.852
```


## Profiling model's performance with SGLang

SGLang provides and automatic way for profiling the e2e model performance. In the docker container run:

```bash
python3 -m sglang.bench_one_batch \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --disable-cuda-graph --mem-fraction-static=0.7 \
    --tp-size 4 --batch 32 --input-len 1024 --output-len 10
```
By changing the different inverse implementations we obtain:

Default triton backend:
```
Prefill. latency: 2.84264 s, throughput:  11527.30 token/s
Decode 0. Batch size: 32, latency: 0.21257 s, throughput:    150.54 token/s
Decode 1. Batch size: 32, latency: 0.21319 s, throughput:    150.10 token/s
Decode 2. Batch size: 32, latency: 0.21345 s, throughput:    149.92 token/s
Decode 3. Batch size: 32, latency: 0.21478 s, throughput:    148.99 token/s
Decode 4. Batch size: 32, latency: 0.21697 s, throughput:    147.49 token/s
Decode.  median latency: 0.21319 s, median throughput:    150.10 token/s
Total. latency:  4.755 s, throughput:   6958.84 token/s
```

Vec column sweep
```
Prefill. latency: 4.19312 s, throughput:   7814.70 token/s
Decode 0. Batch size: 32, latency: 0.20885 s, throughput:    153.22 token/s
Decode 1. Batch size: 32, latency: 0.20610 s, throughput:    155.27 token/s
Decode 2. Batch size: 32, latency: 0.20990 s, throughput:    152.45 token/s
Decode 3. Batch size: 32, latency: 0.20923 s, throughput:    152.94 token/s
Decode 4. Batch size: 32, latency: 0.20734 s, throughput:    154.34 token/s
Decode.  median latency: 0.20901 s, median throughput:    153.11 token/s
Total. latency:  6.073 s, throughput:   5448.66 token/s
```

Rec unroll
```
Prefill. latency: 3.60372 s, throughput:   9092.84 token/s
Decode 0. Batch size: 32, latency: 0.21152 s, throughput:    151.29 token/s
Decode 1. Batch size: 32, latency: 0.20708 s, throughput:    154.53 token/s
Decode 2. Batch size: 32, latency: 0.20664 s, throughput:    154.86 token/s
Decode 3. Batch size: 32, latency: 0.20821 s, throughput:    153.69 token/s
Decode 4. Batch size: 32, latency: 0.20685 s, throughput:    154.70 token/s
Decode.  median latency: 0.20708 s, median throughput:    154.53 token/s
Total. latency:  5.485 s, throughput:   6032.53 token/s
```

# Common errors / FAQ

## RuntimeError: Not enough memory

![image](uploads/c48fca21fc2199cf2b39928951c6dd6a/image.png)

When the available memory is not enough several fixes are possible:
- reduce the --max-num-tokens (1024 always works, but more may be needed)
- increase --mem-fraction-static (0.8)
- reduce/increase the number of NPUs to be used --tp-size (4 or 8)
- check if anyone else is using the NPUs

## Failed to run BiShengIR HIVM pipeline

![image](uploads/ab2120c85e3a430c750bc80a0ced42c5/image.png)

Run:
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/latest/bisheng_toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## RuntimeError: Prefill out of memory

![image](uploads/b0ad41b14672e71c34d7e2bdfcddee38/image.png)

Increase --max-num-tokens (2048 or 4096)

## Triton errors when starting sglang server

On x86-64, you might get errors with Triton, you can try

```bash
pip3 remove triton -y
pip3 install --force-reinstall -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```


# Additional references
https://github.com/sgl-project/sgl-kernel-npu/pull/374
https://open.codehub.huawei.com/innersource/self_spec_infer_G/ascend_operators/files?ref=main&filePath=sglang_profiling%2Fbench_qwen3n.md&isFile=true
