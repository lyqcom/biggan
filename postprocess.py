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
import argparse
import numpy as np
from PIL import Image
from src import inception_utils


def arg_parser():
    parser = argparse.ArgumentParser('proprocess')
    parser.add_argument('--input_path', type=str, default='./preprocess_out/')
    parser.add_argument('--output_path', type=str,
                        default='./postprocess_out/')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    args = arg_parser()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    IS = []
    FID = []
    count = 0
    for b_file in os.listdir(args.input_path):
        f_name = os.path.join(args.input_path, b_file)
        in_data = np.fromfile(f_name, dtype=np.float32)[:49152]
        in_data = in_data.reshape((-1, 3, 128, 128))
        img_mane = b_file.split('.')[0]+'.png'
        outfile = os.path.join(args.output_path, img_mane)
        image = np.transpose(in_data[0], (1, 2, 0))
        im = Image.fromarray(np.uint8(image))
        im.save(outfile)
        print("infer postprogress {}".format(img_mane))
        IS_mean, IS_std = inception_utils.calculate_inception_score(in_data)
        FID_val = inception_utils.cal_fid(in_data, image)
        IS.append(IS_mean)
        FID.append(FID_val)
        count += 1
    print("CAL IS_mean:{} FID:{}".format(round(np.mean(np.array(IS)),3), round(np.mean(np.array(FID)),3)))
    print("postprogress done!")

