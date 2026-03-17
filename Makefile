.PHONY: setup_once install_gh_pto_kernels install_pto_kernels_internal install profile_tri_inv

test_tri_inv:
	python -m pytest -v tests/test_linalg.py

profile_tri_inv:
	bash profiling/run_profiling_tri_inv.sh
	pip install jupyter ipykernel nbconvert pandas seaborn
	export PATH=${PATH}:${HOME}/.local/bin/ && jupyter nbconvert --to notebook --inplace --execute profiling/nbs/plots_tri_inv.ipynb

install:
	export CMAKE_GENERATOR="Unix Makefiles" && . /usr/local/Ascend/ascend-toolkit/set_env.sh && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
