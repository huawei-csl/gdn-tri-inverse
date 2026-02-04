.PHONY: clean setup_once build test

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup_once:
	pip3 install -r requirements.txt
	pip3 install https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp310-cp310-manylinux_2_28_x86_64.whl --index-url https://download.pytorch.org/whl/cpu # https://github.com/sgl-project/sgl-kernel-npu/pull/326
	git clone https://gitcode.com/cann/pto-isa.git PTO_ISA_ROOT

build:
	bash run.sh

test:
	pytest -v tests/
