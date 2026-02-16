.PHONY: setup_once

setup_once:
	pip3 install pyyaml setuptools pytest numpy einops
	pip3 install https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp310-cp310-manylinux_2_28_x86_64.whl --index-url https://download.pytorch.org/whl/cpu # https://github.com/sgl-project/sgl-kernel-npu/pull/326
	pip3 install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir

profile_tri_inv:
	bash profiling/run_profiling_tri_inv.sh
	pip install jupyter ipykernel nbconvert pandas seaborn
	export PATH=${PATH}:${HOME}/.local/bin/ && jupyter nbconvert --to notebook --inplace --execute profiling/nbs/plots_tri_inv.ipynb
