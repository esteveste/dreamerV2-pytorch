import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import common
import math


class RSSM(common.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, discrete=False, act=F.elu,
                 std_act='softplus', min_std=0.1):
        super(RSSM, self).__init__()

        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = getattr(F, act) if isinstance(act, str) else act
        self._std_act = std_act
        self._min_std = min_std

        self._cell = torch.jit.script(GRUCell(self._hidden, self._deter, norm=True))


    def initial(self, batch_size, device):
        '''
        Returns initial RSSM state
        '''

        if self._discrete:
            state = dict(
                logit=torch.zeros(batch_size, self._stoch, self._discrete),
                stoch=torch.zeros(batch_size, self._stoch, self._discrete),
                deter=self._cell.get_initial_state(batch_size))
        else:
            state = dict(
                mean=torch.zeros(batch_size, self._stoch),
                std=torch.zeros(batch_size, self._stoch),
                stoch=torch.zeros(batch_size, self._stoch),
                deter=self._cell.get_initial_state(batch_size))  # dtype

        return common.dict_to_device(state, device)

    def observe(self, embed, action, state=None):
        swap = lambda x: x.permute(1, 0, *list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0], action.device)
        embed, action = swap(embed), swap(action)  # seems to be a permute of batch,sequence to sequence, batch
        post, prior = common.sequence_scan(
            # percorre 1 sequencia de cada vez para o obs step FIXME this is ineficient...but for only 2 layers?
            self.obs_step,
            state, action, embed)
        post = {k: swap(v) for k, v in post.items()}  # it seems to put to (batch,seq) again
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute(1, 0, *list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0], action.device)
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.sequence_scan(self.img_step, state, action)[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        '''
        gets stoch and deter as tensor
        '''

        # FIXME verify shapes of this function
        # stoch = self._cast(state['stoch']) #FIXME cast
        stoch = state['stoch']
        if self._discrete:
            stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, state['deter']], -1)

    def get_dist(self, state):
        '''
        gets the stochastic state distribution
        '''
        if self._discrete:
            logit = state['logit']
            logit = logit.float()  # tf.cast(logit, tf.float32) #fixme cast from fp16
            dist = common.Independent(common.OneHotDist(logit), 1)  # tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            mean = mean.float()  # tf.cast(mean, tf.float32)
            std = std.float()  # tf.cast(std, tf.float32)
            dist = common.Independent(common.Normal(mean, std), 1)  # #tfd.MultivariateNormalDiag(mean, std)
        return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True):
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior['deter'], embed], -1)  # embed is encoder conv output
        x = self._act(self.get('obs_out', nn.Linear, x.shape[-1], self._hidden)(x))
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state['stoch']
        if self._discrete:
            prev_stoch = prev_stoch.reshape(*prev_stoch.shape[:-2], self._stoch * self._discrete)
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._act(self.get('img_in', nn.Linear, x.shape[-1], self._hidden)(x))
        deter = prev_state['deter']
        x, deter = self._cell(x, deter)

        x = self._act(self.get('img_out', nn.Linear, x.shape[-1], self._hidden)(x))
        stats = self._suff_stats_layer('img_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, nn.Linear, x.shape[-1], self._stoch * self._discrete)(x)
            logit = x.reshape(*x.shape[:-1], self._stoch, self._discrete)
            return {'logit': logit}
        else:
            x = self.get(name, nn.Linear, x.shape[-1], 2 * self._stoch)(x)
            mean, std = x.chunk(2, -1)
            std = {
                'softplus': lambda: F.softplus(std),
                'sigmoid': lambda: F.sigmoid(std),
                'sigmoid2': lambda: 2 * F.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        # post prior from observe are dicts
        kld = tdist.kl_divergence
        detach = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)

        free = torch.tensor(free)  # convert to tensor

        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(detach(rhs)))
            value_rhs = kld(self.get_dist(detach(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class ConvEncoder(common.Module):
    def __init__(
            self, depth=32, act=F.elu, kernels=(4, 4, 4, 4), keys=['image']):
        super(ConvEncoder, self).__init__()
        self._act = getattr(F, act) if isinstance(act, str) else act
        self._depth = depth
        self._kernels = kernels
        self._keys = keys

    def forward(self, obs):  # obs shape(batch?),W,H,C, can be torch object
        if tuple(self._keys) == ('image',):
            # from B,T,C,H,W or B,C,H,W to (B*T,C,H,W)
            x = torch.reshape(obs['image'], (-1, *obs['image'].shape[-3:])).to(
                memory_format=torch.channels_last)  # FIXME testing
            for i, kernel in enumerate(self._kernels):
                depth = 2 ** i * self._depth
                x = self._act(self.get(f'h{i}', nn.Conv2d, x.shape[1], depth, kernel, stride=2)(x))

            # permute only needed to be equal to tf, linear layers dont need order
            # return x.permute(0, 2, 3, 1).reshape(*obs['image'].shape[:-3], -1)
            return x.reshape(*obs['image'].shape[:-3], -1)  # to B,T,W*H*C or B,W*H*C

        else:
            raise NotImplementedError("ConvEncoder - not 'image' key",self._keys)
            # FIXME DO Later
            # # dtype = prec.global_policy().compute_dtype
            # features = []
            # for key in self._keys:
            #     value = tf.convert_to_tensor(obs[key])
            #     if value.dtype.is_integer:
            #         value = tf.cast(value, dtype)
            #         semilog = tf.sign(value) * tf.math.log(1 + tf.abs(value))
            #         features.append(semilog[..., None])
            #     elif len(obs[key].shape) >= 4:
            #         x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
            #         for i, kernel in enumerate(self._kernels):
            #             depth = 2 ** i * self._depth
            #             x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
            #         x = tf.reshape(x, [x.shape[0], int(np.prod(x.shape[1:]))])
            #         shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
            #         features.append(tf.reshape(x, shape))
            #     else:
            #         raise NotImplementedError((key, value.dtype, value.shape))
            # return tf.concat(features, -1)


class ConvDecoder(common.Module):
    def __init__(self, shape=(3, 64, 64), depth=32, act=F.elu, kernels=(5, 5, 6, 6)):
        super(ConvDecoder, self).__init__()
        self._shape = shape
        self._depth = depth
        self._act = getattr(F, act) if isinstance(act, str) else act
        self._kernels = kernels

    def forward(self, features):
        ConvT = nn.ConvTranspose2d
        x = self.get('hin', nn.Linear, features.shape[-1], 32 * self._depth)(features)
        x = x.reshape(-1, 32 * self._depth, 1, 1).to(
            memory_format=torch.channels_last)  # FIXME testing  # 32*depth, 1,1
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            if i == len(self._kernels) - 1:  # last one
                depth = self._shape[0]
                act = lambda x: x
            x = self.get(f'h{i}', ConvT, x.shape[1], depth, kernel, stride=2)(x)
            x = act(x)
        # this permute is essential for the final shape IS NOT NOW
        # mean = x.permute(0, 2, 3, 1).reshape(*features.shape[:-1], *self._shape)

        mean = x.reshape(*features.shape[:-1], *self._shape)  # back to shape (magic,C,H,W)
        return common.Independent(common.Normal(mean, 1), len(self._shape))


class MLP(common.Module):

    def __init__(self, shape, layers, units, act=F.elu, **out):
        '''
        :param shape: final shape for DistLayer
        :param layers: nr layers
        :param units: hidden units
        :param act:
        :param out: keyargs to DistLayer
        '''
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._act = getattr(F, act) if isinstance(act, str) else act
        self._out = out

    def forward(self, features):
        x = features

        for index in range(self._layers):
            x = self.get(f'h{index}', nn.Linear, x.shape[-1], self._units)(x)
            x = self._act(x)
        return self.get('out', DistLayer, self._shape, **self._out)(x)


#
class GRUCell(nn.Module):
    # only 1 layer
    def __init__(self, input_size, size, norm=False, act=torch.tanh, update_bias=-1):
        # DISABLED kwargs for now
        super().__init__()
        self._size = size
        self._act = getattr(F, act) if isinstance(act, str) else act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size + self._size, 3 * self._size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-3)  # eps equal to tf

    #     self.reset_parameters()
    #
    # def reset_parameters(self): #IF LINEAR
    #     std = 1.0 / math.sqrt(self.hidden_size)
    #     for w in self.parameters():
    #         w.data.uniform_(-std, std)
    #

    @property
    def state_size(self):
        return self._size

    @torch.jit.export
    def get_initial_state(self, batch_size: int):  # defined by tf.keras.layers.AbstractRNNCell
        return torch.zeros(batch_size, self._size)

    def forward(self, input, state):

        cat_input = torch.cat([input, state], -1)
        parts = self._layer(cat_input)

        if self._norm is not False:  # check if jit compatible
            #     dtype = parts.dtype
            #     parts = tf.cast(parts, tf.float32)
            #     parts = self._norm(parts)
            #     parts = tf.cast(parts, dtype)

            parts = self._norm(parts)

        reset, cand, update = parts.chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)  # it also multiplies the reset by the input
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, output


class DistLayer(common.Module):
    # Only called in MLP class

    def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
        super(DistLayer, self).__init__()
        self._shape = shape  # shape can be [], its equivalent to 1.0 in np.prod
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get('out', nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
        out = out.reshape([*inputs.shape[:-1],
                           *self._shape])  # FIXME caution with future shape orders, supostamente (batchsize, *_shape)
        # out = tf.cast(out, tf.float32) #FIXME
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self.get('std', nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
            std = std.reshape([*inputs.shape[:-1], *self._shape])  # FIXME caution with future shape orders
            # std = tf.cast(std, tf.float32) #FIXME
        if self._dist == 'mse':
            dist = common.Normal(out, 1.0)
            return common.Independent(dist, len(self._shape))
        if self._dist == 'normal':
            raise NotImplementedError(self._dist)
            # NOTE Doesnt make sense std to be negative
            # NOT USED in algorightm

            # print(out.shape,std.shape)
            # dist = tdist.Normal(out, std) #Fixme std can only be positive
            # return tdist.Independent(dist, len(self._shape))
        if self._dist == 'binary':
            dist = common.Bernoulli(logits=out, validate_args=False)  # FIXME no validate_args
            return common.Independent(dist, len(self._shape))
        if self._dist == 'tanh_normal':
            raise NotImplementedError(self._dist)  # FIXME ERROR too big for now

            # mean = 5 * torch.tanh(out / 5)
            # std = F.softplus(std + self._init_std) + self._min_std
            # dist = tdist.Normal(mean, std)
            # dist = tdist.TransformedDistribution(dist, common.TanhBijector()) #tfd.TransformedDistribution(dist, common.TanhBijector())
            # dist = tdist.Independent(dist, len(self._shape))
            # return common.SampleDist(dist)
        if self._dist == 'trunc_normal':
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std

            dist = common.TruncNormalDist(torch.tanh(out), std, -1, 1)
            return common.Independent(dist, 1)
        if self._dist == 'onehot':
            dist = common.OneHotDist(logits=out)
            dist.orig_logits = out

            return dist
        NotImplementedError(self._dist)
