import pathlib
import pickle
import re
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torch.optim as optim

import common
import elements.logger

try:
    import wandb
except Exception:
    wandb = None #dont force wandb dependence

class Module(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    # get function from original Dreamer
    # in the end this should be a traced function, where ifs are ignored
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            raise Exception("Pytorch Module already has this")
            # self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)

            #default pytorch - kaimain uniform

            # # all with glorot uniform (no apparent gain) and zeros init bias todo gain?
            if hasattr(self._modules[name],"weight"):
                torch.nn.init.xavier_uniform_(self._modules[name].weight)
                if self._modules[name].bias.data is not None:
                    # self._modules[name].bias.data.zero_()
                    torch.nn.init.zeros_(self._modules[name].bias)

            if ctor in [nn.Conv2d, nn.ConvTranspose2d]:
                print("setting memory format to channels last")
                self._modules[name] = self._modules[name].to(memory_format=torch.channels_last) #FIXME testing

        return self._modules[name]


    # for reptile based on https://github.com/gabrielhuang/reptile-pytorch
    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = torch.autograd.Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = torch.autograd.Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class EmptyOptimizer():
    def backward(self, loss, retain_graph=False):
        pass

    def step(self, loss):
        return {}


class Optimizer():
    def __init__(
            self, name, modules, lr, eps=1e-4, clip=None, wd=None, opt='adam'):
        # assert 0 <= wd < 1 #FIXME
        # assert not clip or 1 <= clip
        if isinstance(modules, list):  # then is a list a parameters
            modules = itertools.chain(*modules)
        self._name = name
        self._clip = clip
        self._wd = wd
        # self._wd_pattern = wd_pattern # FIXME IGNORE FOR NOW PATTERNS, they are applied to all
        self._opt = {
            'adam_tf': lambda p: Adam_tf(p, lr, eps=eps, weight_decay=wd),
            'adam': lambda p: optim.Adam(p,lr, eps=eps, weight_decay=wd),
            'adamw': lambda p: optim.AdamW(p, lr, eps=eps,weight_decay=wd),
            'adamw_tf': lambda p: AdamW_tf(p, lr, eps=eps, weight_decay=wd),
            'adamax': lambda p: optim.Adamax(p, lr, eps=eps, weight_decay=wd),
            'sgd': lambda p: optim.SGD(p, lr, weight_decay=wd),
            'momentum': lambda p: optim.SGD(p, lr, momentum=0.9, weight_decay=wd),
        }[opt](modules)

        self._scaler = torch.cuda.amp.GradScaler(enabled=common.ENABLE_FP16)

        print(f'Init optimizer - {self._name}')

    def backward(self, loss, retain_graph=False):
        self._scaler.scale(loss).backward(retain_graph=retain_graph)

    def step(self, loss):
        metrics = {}
        metrics[f'{self._name}_loss'] = loss.item()

        if self._clip:
            # gets unscaled gradients
            self._scaler.unscale_(self._opt)

            # Assuming only 1 optimizer group of tensors
            norm = torch.nn.utils.clip_grad_norm_(self._opt.param_groups[0]['params'], self._clip,
                                                  error_if_nonfinite=False)
            metrics[f'{self._name}_grad_norm'] = norm.item()  # implementation not equal to tf

        self._scaler.step(self._opt)
        self._scaler.update()

        # opt.zero_grad()  # set_to_none=True here can modestly improve performance

        if common.ENABLE_FP16 and common.DEBUG_METRICS:
            metrics[f'{self._name}_loss_scale'] = self._scaler.get_scale() #incurs a CPU-GPU sync. todo optimization

        return metrics

class TensorBoardOutputPytorch:

    ## FIXME image dataformats='CHW' by default

    def __init__(self, logdir, fps=20):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(logdir), max_queue=1000)
        self._fps = fps

    def __call__(self, summaries):
        for step, name, value in summaries:
            if len(value.shape) == 0:
                self._writer.add_scalar('scalars/' + name, value, step)
            elif len(value.shape) == 2:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 3:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 4:
                self._video_summary(name, value, step)
                # self._writer.add_video(name,value[None], step,fps=self._fps)
                #vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
                # batch, time, channels, height and width
        self._writer.flush()

    def _video_summary(self, name, video, step):
        # import tensorflow as tf
        # import tensorflow.compat.v1 as tf1
        from tensorboard.compat.proto.summary_pb2 import Summary


        name = name if isinstance(name, str) else name.decode('utf-8')
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            # video = video.transpose((0,3,1,2))
            T, H, W, C = video.shape
            # summary = tb.RecordWriter()
            image = Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = elements.logger.encode_gif(video, self._fps)
            self._writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=name, image=image)]),step)
            # tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print('GIF summaries require ffmpeg in $PATH.', e)
            self._writer.add_image(name, video, step)


class WandBOutput:

    ## FIXME image dataformats='CHW' by default

    def __init__(self, fps=20,**kwargs):
        assert wandb, 'make sure you have wandb installed'

        self._fps = fps
        wandb.init(**kwargs)

    def __call__(self, summaries):
        for step, name, value in summaries:
            if len(value.shape) == 0:
                wandb.log({'scalars/' + name: value}, step=step)
            elif len(value.shape) == 2 or len(value.shape) == 3:
                wandb.log({name: wandb.Image(value)},step=step)
            elif len(value.shape) == 4:
                name = name if isinstance(name, str) else name.decode('utf-8')
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = value.transpose((0, 3, 1,2))
                wandb.log({name: wandb.Video(value, fps=self._fps, format='mp4')},step=step)


#### custom stuff

import math
import torch
from torch.optim.optimizer import Optimizer as torch_Optimizer



class Adam_tf(torch_Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_tf, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_tf, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps']) / math.sqrt(bias_correction2)

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamW_tf(torch_Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW_tf, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW_tf, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps']) # we add eps under the denominator
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
