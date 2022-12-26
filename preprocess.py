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
# coding=utf-8

import os
import shutil
import argparse
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser('preprocess')
    parser.add_argument('--output_path', type=str, default='./preprocess_out/')
    parser.add_argument('--img_path', type=str, default='./preprocess_out/')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    args = arg_parser()
    input_data_path = os.path.join(args.output_path, "input_data")
    label_path = os.path.join(args.output_path, "label")

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
        os.makedirs(input_data_path)
        os.makedirs(label_path)
    else:
        os.makedirs(input_data_path)
        os.makedirs(label_path)

    input_data = np.random.randn(1, 3, 128, 128).astype(np.float32)
    label_data = np.zeros((1000, 1)).astype(np.float32)
    file_name = "bigan" + ".bin"
    latent_code_file_path = os.path.join(input_data_path, file_name)
    label_file_path = os.path.join(label_path, file_name)
    input_data.tofile(latent_code_file_path)
    label_data.tofile(label_file_path)
    print("bin files done!")

    path = args.img_path
    new_path = './dataset'
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        os.makedirs(new_path)
    else:
        os.makedirs(new_path)
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            if files[i][-4:] == 'JPEG':
                file_path = root+'/'+files[i]
                print("get file_path", file_path)
                new_file_path = new_path + '/' + files[i]
                shutil.copy(file_path, new_file_path)
