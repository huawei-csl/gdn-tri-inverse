.PHONY: clean test_tri_inv install profile_tri_inv profile_tilelang_full_gdn

clean:
	rm -rf build kernel_meta/ output/

test_tri_inv:
	python -m pytest -v tests/test_linalg.py

profile_tri_inv:
	bash profiling/run_profiling_tri_inv.sh
	pip install jupyter ipykernel nbconvert pandas seaborn
	export PATH=${PATH}:${HOME}/.local/bin/ && jupyter nbconvert --to notebook --inplace --execute profiling/nbs/plots_tri_inv.ipynb

install:
	export CMAKE_GENERATOR="Unix Makefiles" && . /usr/local/Ascend/ascend-toolkit/set_env.sh && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
