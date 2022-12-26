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
import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model import BigGAN as model
from src import utils
import cv2
from PIL import Image


def arg_parser():
    parser = argparse.ArgumentParser('proprocess')
    parser.add_argument('--input_data', type=str, default='./preprocess_in/')
    parser.add_argument('--output_path', type=str,
                        default='./saved_path/')
    parser.add_argument('--used_ckpt', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    context.set_context(device_id=args.device_id,
                        mode=context.GRAPH_MODE, device_target="Ascend")
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    img_in = cv2.imread(args.input_data)
    img_in = cv2.resize(img_in, (128, 128), interpolation=cv2.INTER_CUBIC)
    input_data = Tensor(np.random.randn(1, 3, 128, 128),
                        dtype=mindspore.float32)
    model_G = model.Generator(100, 3)
    param_G = load_checkpoint(args.used_ckpt)
    load_param_into_net(model_G, param_G)
    gan_img = model_G(input_data)
    img = gan_img.asnumpy().flatten()[0:49152].reshape(128, 128, 3)/127.5
    img = img + img_in
    saved_file = os.path.join(args.output_path, 'gan_img.jpg')
    cv2.imwrite(saved_file, img)
    IS_Val, _ = inception_utils.calculate_inception_score(img,iter_num=1)
    FID_val = inception_utils.CAL_FID(input_data, gan_img, -1)
    print("model eval done, check img at {}, Get IS_Val:{} FID_val:{}".format(saved_file, round(IS_Val, 3),round(FID_val, 3)))
