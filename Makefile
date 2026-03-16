.PHONY: setup_once install_gh_pto_kernels setup_once profile_tri_inv install


install_gh_pto_kernels:
	export CMAKE_GENERATOR="Unix Makefiles" && pip3 install --force-reinstall -v git+https://github.com/huawei-csl/pto-kernels.git --extra-index-url https://download.pytorch.org/whl/cpu

setup_once:
	pip3 install pyyaml setuptools pytest numpy einops
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu # https://github.com/sgl-project/sgl-kernel-npu/pull/326
	pip3 install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir

profile_tri_inv:
	bash profiling/run_profiling_tri_inv.sh
	pip install jupyter ipykernel nbconvert pandas seaborn
	export PATH=${PATH}:${HOME}/.local/bin/ && jupyter nbconvert --to notebook --inplace --execute profiling/nbs/plots_tri_inv.ipynb


install: install_gh_pto_kernels
	export CMAKE_GENERATOR="Unix Makefiles" && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
