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
"""for trans model .ckpt to mindir"""
import numpy as np
import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from model.BigGAN import Generator
from src import utils
# pylint: disable=W0703


def main():
    parser_ = utils.prepare_parser()
    config = vars(parser_.parse_args())
    try:
        mindspore.context.set_context(device_id=int(config['use_device']))
        # set graph mode device_target="Ascend"
        mindspore.context.set_context(
            mode=mindspore.context.GRAPH_MODE, device_target="Ascend")
    except Exception as e:
        print(str(e))
        mindspore.context.set_context(
            mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    model_G = Generator(**config)
    construct_data = mindspore.Tensor(np.random.randn(
        1, 3, 128, 128), dtype=mindspore.float32)
    param_G = load_checkpoint(config['ckpt_pth'])
    load_param_into_net(model_G, param_G)
    model_G.set_train(False)
    export(model_G, construct_data, file_name="biggan", file_format="MINDIR")
    print("model exported done")


if __name__ == '__main__':
    main()
