import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from dreamerv2 import agent
import common


class Random(common.Module):

    def __init__(self, action_space):
        super(Random, self).__init__()
        self._action_space = action_space

    def actor(self, feat):
        shape = feat.shape[:-1] + [self._action_space.shape[-1]]
        if hasattr(self._action_space, 'n'):
            return common.OneHotDist(torch.zeros(shape))
        else:
            dist = torch.Uniform(-torch.ones(shape), torch.ones(shape))
            return torch.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(common.Module):
    pass
    # FIXME
    # def __init__(self, config, world_model, num_actions, step, reward=None):
    #     super(Plan2Explore, self).__init__()
    #     self.config = config
    #     self.reward = reward
    #     self.wm = world_model
    #     self.ac = agent.ActorCritic(config, step, num_actions)
    #     self.actor = self.ac.actor
    #     stoch_size = config.rssm.stoch
    #     if config.rssm.discrete:
    #         stoch_size *= config.rssm.discrete
    #     size = {
    #         'embed': 32 * config.encoder.depth,
    #         'stoch': stoch_size,
    #         'deter': config.rssm.deter,
    #         'feat': config.rssm.stoch + config.rssm.deter,
    #     }[self.config.disag_target]
    #     self._networks = [
    #         common.MLP(size, **config.expl_head)
    #         for _ in range(config.disag_models)]
    #     self.opt = common.Optimizer('expl', **config.expl_opt)
    #
    # def train(self, start, context, data):
    #     metrics = {}
    #     stoch = start['stoch']
    #     if self.config.rssm.discrete:
    #         stoch = stoch.reshape(*stoch.shape[:-2], stoch.shape[-2] * stoch.shape[-1])
    #     target = {
    #         'embed': context['embed'],
    #         'stoch': stoch,
    #         'deter': start['deter'],
    #         'feat': context['feat'],
    #     }[self.config.disag_target]
    #     inputs = context['feat']
    #     if self.config.disag_action_cond:
    #         # action = tf.cast(data['action'], inputs.dtype)
    #         action = data['action']  # FIXME cast
    #         inputs = torch.cat([inputs, action], -1)
    #     metrics.update(self._train_ensemble(inputs, target))
    #     metrics.update(self.ac.train(self.wm, start, self._intr_reward))
    #     return None, metrics
    #
    # def _intr_reward(self, feat, state, action):
    #     inputs = feat
    #     if self.config.disag_action_cond:
    #         # action = tf.cast(action, inputs.dtype) #FIxme Cast
    #         inputs = torch.cat([inputs, action], -1)
    #     preds = [head(inputs).mean() for head in self._networks]
    #     disag = torch.tensor(preds).std(0).mean(-1)
    #     if self.config.disag_log:
    #         disag = torch.log(disag)
    #     reward = self.config.expl_intr_scale * disag
    #     if self.config.expl_extr_scale:
    #         reward += self.config.expl_extr_scale * self.reward(feat, state, action)
    #     return reward
    #
    # def _train_ensemble(self, inputs, targets):
    #     if self.config.disag_offset:
    #         targets = targets[:, self.config.disag_offset:]
    #         inputs = inputs[:, :-self.config.disag_offset]
    #     targets = targets.detach()
    #     inputs = inputs.detach()
    #     # with tf.GradientTape() as tape:
    #     preds = [head(inputs) for head in self._networks]
    #     loss = -sum([pred.log_prob(targets).mean() for pred in preds])
    #     # metrics = self.opt(tape, loss, self._networks)
    #     metrics = {"loss": loss}  # FIXME metrics
    #     return metrics


class ModelLoss(common.Module):
    # FIXME

    def __init__(self, config, world_model, num_actions, step, reward=None):
        super(ModelLoss, self).__init__()
        self.config = config
        self.reward = reward
        self.wm = world_model
        self.ac = agent.ActorCritic(config, step, num_actions)
        self.actor = self.ac.actor
        self.head = common.MLP([], **self.config.expl_head)
        self.opt = common.Optimizer('expl', **self.config.expl_opt)

        raise NotImplementedError

    # def train(self, start, context, data):
    #     metrics = {}
    #     # target = tf.cast(context[self.config.expl_model_loss], tf.float32)
    #     target = context[self.config.expl_model_loss]  # Fixme Cast
    #     # with tf.GradientTape() as tape:
    #     loss = -self.head(context['feat']).log_prob(target).mean()
    #     # metrics.update(self.opt(tape, loss, self.head)) # FIXME  optim
    #     metrics.update(self.ac.train(self.wm, start, self._intr_reward))
    #     return None, metrics
    #
    # def _intr_reward(self, feat, state, action):
    #     reward = self.config.expl_intr_scale * self.head(feat).mode()
    #     if self.config.expl_extr_scale:
    #         reward += self.config.expl_extr_scale * self.reward(feat, state, action)
    #     return reward
