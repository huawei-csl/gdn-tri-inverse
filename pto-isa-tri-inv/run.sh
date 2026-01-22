#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

#example script

AARCH=$(uname -i)
PY_VER=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("SOABI").split("-")[1])')

set -e 
export ASCEND_HOME_PATH=/usr/local/Ascend/
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# specify your path to pto-isa or use ``make setup_once`
export PTO_LIB_PATH="$(pwd)/PTO_ISA_ROOT"

rm -fr build op_extension.egg-info
python3 setup.py bdist_wheel 

pushd dist || exit 1
python3 -m pip install --force-reinstall "op_extension-0.0.0-cp${PY_VER}-cp${PY_VER}-linux_${AARCH}.whl"
popd

pushd test || exit 1
python3 test.py
popd
