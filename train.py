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
# pylint: disable=W0702
# pylint: disable=C0121
# pylint: disable=C0411

import sys
from model import BigGAN as model
from src.utils import GenWithLossCell, DisWithLossCell
from src import inception_utils
from src import utils
from model import losses
import mindspore
import numpy as np
import time
sys.path.append('../')


def run(config):
    try:
        mindspore.context.set_context(device_id=int(config['use_device']))
        # set graph mode device_target="Ascend"
        mindspore.context.set_context(
            mode=mindspore.context.GRAPH_MODE, device_target="Ascend")  # PYNATIVE_MODE or GRAPH_MODE
    except:
        mindspore.context.set_context(
            mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    if config['distributed'] == True:
        mindspore.communication.init()
        mindspore.context.set_auto_parallel_context(gradients_mean=True,
                                                    device_num=8,
                                                    parallel_mode=mindspore.context.ParallelMode.DATA_PARALLEL)

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    utils.seed_rng(config['seed'])
    utils.prepare_root(config)
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    G = model.Generator(**config)
    D = model.Discriminator(**config)

    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True})
    else:
        G_ema = None

    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
    GD = model.G_D(G, D)
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = mindspore.nn.DataParallel(GD)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})
    loaders = loaders.batch(config['batch_size'], drop_remainder=True)
    netG_with_loss = GenWithLossCell(G, D)
    netD_with_loss = DisWithLossCell(G, D)
    G_lr = 5e-5
    D_lr = 2e-4
    optimizerG = mindspore.nn.Adam(
        params=G.trainable_params(), learning_rate=G_lr)
    optimizerD = mindspore.nn.Adam(
        params=D.trainable_params(), learning_rate=D_lr)
    model_train = utils.TrainOneStepCell(netG_with_loss,
                                         netD_with_loss,
                                         optimizerG,
                                         optimizerD)
    G.set_train()
    D.set_train()
    reshape_ = mindspore.ops.Reshape()
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print("Beginning epoch:{} ".format(epoch))
        for i, (x, y) in enumerate(loaders):
            time_start = time.time()
            if config['D_fp16']:
                x = mindspore.ops.Cast(x, mindspore.float16)
            x = reshape_(x, (config['batch_size'], 3, 128, 128))
            y = mindspore.Tensor(y, mindspore.float32)
            x_ = mindspore.Tensor(np.random.randn(
                config['batch_size'], 3, 128, 128), dtype=mindspore.float32)
            dout, gout = model_train(x, x_, y)
            time_end = time.time()
            if (i+1) % 200 == 0:
                img_g = G(x_)
                IS_mean, IS_std = inception_utils.calculate_inception_score(
                    img_g, iter_num=epoch)
                FID = inception_utils.CAL_FID(x, img_g, epoch)
                gout = losses.generator_loss(gout, epoch)
                dout = losses.discriminator_loss(dout)
                print("per step time:{} ms".format(round((time_end-time_start)*1000,3)))
                print("training iter: {} Loss:{}".format(i, gout+dout))
                print("Get IS_mean:{}, IS_std:{}, FID:{}".format(
                    round(IS_mean, 3), round(IS_std, 3), round(FID, 3)))
            if i >= config['data_thr']:
                break
        #for save most five weights 
        if epoch % 19 == 4:
            utils.save_weights(G, D, epoch)
        state_dict['epoch'] += 1


def main():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print("**"*10)
    print(config)
    run(config)


if __name__ == '__main__':
    main()
