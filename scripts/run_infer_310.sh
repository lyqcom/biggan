#!/bin/bash
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

gen_model=$1
data_path=$2

device_id=0

echo "generator mindir name: "$gen_model
echo "dataset path: "$data_path
echo "device id: "$device_id
export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

preprocess_out='./preprocess_out/'
echo "Start to preprocess attr file..."
python preprocess.py --output_path=$preprocess_out --img_path=$data_path &> preprocess.log
echo "Attribute file generates successfully!"

#compile_app
echo "Start to compile source code..."
cd ./ascend310_infer || exit
bash build.sh &> build.log
echo "Compile successfully."

if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

#infer
cd - || exit
if [ -d result_Files ]; then
    rm -rf ./result_Files
fi
if [ -d time_Result ]; then
    rm -rf ./time_Result
fi
mkdir result_Files
mkdir time_Result
echo "Start to execute inference..."
./ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path='./dataset' --device_id=$device_id --image_height=128 --image_width=128 &> infer.log


if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi

#postprocess_data

echo "Start to postprocess image file..."
if [ -d postprocess_out ]; then
    rm -rf ./postprocess_out
fi
python postprocess.py --input_path="./result_Files/" --output_path="./postprocess_out/" &> postprocess.log

rm -rf ./result_Files
rm -rf ./postprocess_out

if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi
