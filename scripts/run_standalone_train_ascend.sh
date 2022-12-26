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

if [ -e "nohup.out" ]; then
  rm -f nohup.out
fi
cur_path=`pwd`
DEVICE_ID=$1
DATASETS_PATH=$2

export RANK_TABLE_FILE=${cur_path}/hccl_2p.json
echo $RANK_TABLE_FILE
#export RANK_SIZE=2
nohup python3 -u train.py  \
--use_device $DEVICE_ID \
--distributed False \
--datasets_path $DATASETS_PATH &