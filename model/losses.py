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
import mindspore
import numpy as np


def loss_hinge_dis(dis_real):
    mean_ = mindspore.ops.ReduceMean()
    relu_ = mindspore.ops.ReLU()
    loss_real = mean_(relu_(1. + dis_real))
    return loss_real


def loss_hinge_gen(gen_real, step):
    reducemean = mindspore.ops.ReduceMean(keep_dims=True)
    loss = reducemean(gen_real)
    return loss


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
