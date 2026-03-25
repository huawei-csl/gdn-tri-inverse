.PHONY: clean test_tri_inv install profile_tri_inv profile_gdn_layer profile_tilelang_full_gdn analyze_tilelang_opt_solve_tril

clean:
	rm -rf build kernel_meta/ output/
test_tri_inv:
	python -m pytest -v tests/test_linalg.py

profile_tri_inv:
	bash profiling/run_profiling_tri_inv.sh
	pip install jupyter ipykernel nbconvert pandas seaborn
	export PATH=${PATH}:${HOME}/.local/bin/ && jupyter nbconvert --to notebook --inplace --execute profiling/nbs/plots_tri_inv.ipynb

profile_gdn_layer:
	bash profiling/run_profiling_gdn.sh

profile_tilelang_full_gdn: # Profile the tilelang-ascend GDN layer break-down stacked plot
	python profiling/bench_tilelang_full_gdn.py

analyze_tilelang_opt_solve_tril: # Analyze the numerical accuracy of the tilelang-ascend optimized solve_tril implementation
	python profiling/analysis_tilelang_opt_solve_tril.py

install:
	export CMAKE_GENERATOR="Unix Makefiles" && . /usr/local/Ascend/ascend-toolkit/set_env.sh && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
