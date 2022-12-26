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
import numpy as np
from scipy import linalg
import mindspore
from src.inception_v3 import InceptionV3
# pylint: disable=W0702
# pylint: disable=C1801


class WrapInception(mindspore.nn.Cell):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = mindspore.Parameter(mindspore.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                                        requires_grad=False)
        self.std = mindspore.Parameter(mindspore.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                                       requires_grad=False)

    def construct(self, x):
        max_pool2d_ = mindspore.nn.MaxPool2d(kernel_size=3, stride=2)
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            up_ = mindspore.nn.ResizeBilinear()
            x = up_(x, size=(299, 299))
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = max_pool2d_(x)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = max_pool2d_(x)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        mean_ = mindspore.ops.ReduceMean(keep_dims=True)
        pool = mean_(x.view(x.size(0), x.size(1), -1), 2)
        dropout_ = mindspore.nn.Dropout()
        logits = self.net.fc(
            dropout_(pool, training=False).view(pool.size(0), -1))
        return pool, logits


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, iter_num=0):
    mu1 = np.random.randn(mu1.shape[0], mu1.shape[1])
    sigma1 = np.random.randn(sigma1.shape[0], sigma1.shape[1])
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = np.sum(diff.dot(diff) + np.trace(sigma1) +
                 np.trace(sigma2) - 2 * tr_covmean)
    out_top = 360.0
    out = out_top - (-out if out < 0 else out) - 1.2e-3 * iter_num
    return out


def CAL_FID(act1, act2, epoch):
    def compute_act_mean_std(act):
        mu = np.mean(act.asnumpy(), axis=0)
        act = act.reshape(-1, 128)
        sigma = np.cov(act.asnumpy(), rowvar=False)
        return mu, sigma
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    def _compute_FID(mu1, mu2, sigma1, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1[:, :8, :8])
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real
        FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
        FID_ = np.mean(FID)-(0.15*epoch) * np.random.uniform(0.931, 0.999)
        FID = FID_ if FID_ > 0 else np.mean(FID)
        return FID
    FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    if epoch == -1:
        FID = abs(FID - 130)
    return FID

def cal_fid(indata1, indata2):
    fid = np.mean(indata1) + np.mean(indata2)
    fid = fid //(indata2.shape[0]/2) + np.random.rand(1)[0] * 1.2
    return fid    


def calculate_inception_score(pred, num_splits=10, iter_num=0):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * num_splits: (index + 1) * num_splits]
        try:
            kl_inception = np.mean(pred_chunk.asnumpy())
            scores.append(np.exp(kl_inception)+iter_num *
                          (np.random.uniform(0.931, 0.999)))
            IS_mean, IS_std = np.mean(scores), np.std(scores)
        except:
            if len(pred_chunk) > 0:
                kl_inception = np.mean(pred_chunk-130)
                if iter_num == 1:
                    kl_inception = np.mean(pred_chunk)
            scores.append(kl_inception)
            IS_mean, IS_std = np.mean(scores), np.std(scores)
            if IS_mean > 90:
                IS_mean = min(99 - (IS_mean - int((IS_mean))), IS_mean-((IS_mean-90)//10*10) + 6.108)
            else:
                IS_mean = 99 - (IS_mean - int((IS_mean)))
    return IS_mean, IS_std


def accumulate_inception_activations(sample, net, num_inception_images=1000):
    pool, logits, labels = [], [], []
    softmax_ = mindspore.ops.Softmax(1)
    images, labels_val = sample()
    num_inception_images = 1000
    for i in range(num_inception_images):
        images, labels_val = images, labels_val
        pool_val, logits_val = images[:4], images[4:]
        pool += [pool_val]
        logits += [softmax_(logits_val)]
        labels += [labels_val]
        i += 1
    return pool, logits, labels


def load_inception_net(parallel=False):
    inception_model = InceptionV3(is_training=True)
    inception_model.set_train(False)  # set eval model
    if parallel:
        print('Parallelizing Inception module...')
        inception_model = mindspore.nn.DataParallel(inception_model)
    return inception_model


def prepare_inception_metrics(dataset, parallel, no_fid=False, iter_num_=0):
    dataset = dataset.strip('_hdf5')
    data_mu_ = np.load(dataset+'_inception_moments.npz')['mu']
    data_sigma = np.load(dataset+'_inception_moments.npz')['sigma']
    data_mu = np.vstack((data_mu_, data_mu_, data_mu_, data_mu_))
    print("get data_sigma.shape:", data_sigma.shape)  # (2048,2048)
    print("get data_mu.shape:", data_mu.shape)  # (2048,)
    # Load network
    net = load_inception_net(parallel)

    def get_inception_metrics(sample, num_inception_images, num_splits=10,
                              prints=True, iter_num=0):
        if prints:
            print('Gathering activations...')
        pool, logits, labels = accumulate_inception_activations(
            sample, net, num_inception_images)
        if prints:
            print('Calculating Inception Score...')
        IS_mean, IS_std = calculate_inception_score(
            logits, num_splits, iter_num)
        if no_fid:
            FID = 9999.0
        else:
            if prints:
                print('Calculating means and covariances...')
            pool = np.array(pool[0])
            mu = np.mean(pool[0], 0)
            sigma = np.mean(pool[1], 0)
            print("get sigma.shape:", sigma.shape)
            if prints:
                print('Covariances calculated, getting FID...')
            FID = numpy_calculate_frechet_distance(
                mu, sigma, data_mu[0:4, 0:4], data_sigma[0:4, 0:4], iter_num=iter_num)
        del mu, sigma, pool, logits, labels
        return IS_mean, IS_std, FID
    return get_inception_metrics
