import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import elements
import common

from dreamerv2 import expl


class Agent(common.Module):

    def __init__(self, config, logger, actspce, step):  # , dataset):
        super(Agent, self).__init__()
        self.config = config
        self._logger = logger  # not used
        self._action_space = actspce
        self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        self.step = step

        self.wm = WorldModel(self.step, config)
        self._task_behavior = ActorCritic(config, self.step, self._num_act)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(actspce),
            plan2explore=lambda: expl.Plan2Explore(
                config, self.wm, self._num_act, self.step, reward),
            model_loss=lambda: expl.ModelLoss(
                config, self.wm, self._num_act, self.step, reward),
        )[config.expl_behavior]()

        # init modules without optimizers (once in opt)
        with torch.no_grad():
            channels = 1 if config.grayscale else 3
            self.train({'image': torch.rand(1, 4, channels, *config.image_size),
                        'action': torch.rand(1, 4, self._num_act),
                        'reward': torch.rand(1, 4),
                        'discount': torch.rand(1, 4),
                        'done': torch.rand(1, 4)})

    def policy(self, obs, state=None, mode='train'):
        self.step = self._counter

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):

                if state is None:
                    latent = self.wm.rssm.initial(len(obs['image']), obs['image'].device)
                    action = torch.zeros(len(obs['image']), self._num_act).to(obs['image'].device)
                    state = latent, action
                elif obs['reset'].any():
                    # conversion
                    # state = tf.nest.map_structure(lambda x: x * common.pad_dims(
                    #     1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)

                    latent, action = state
                    latent = {k: v * common.pad_dims(1.0 - obs['reset'], len(v.shape)) for k, v in latent.items()}
                    action = action * common.pad_dims(1.0 - obs['reset'], len(action.shape))
                    state = latent, action

                latent, action = state
                embed = self.wm.encoder(self.wm.preprocess(obs))
                sample = (mode == 'train') or not self.config.eval_state_mean
                latent, _ = self.wm.rssm.obs_step(latent, action, embed, sample)
                feat = self.wm.rssm.get_feat(latent)
                if mode == 'eval':
                    actor = self._task_behavior.actor(feat)
                    action = actor.mode
                elif self._should_expl(self.step):
                    actor = self._expl_behavior.actor(feat)
                    action = actor.sample()
                else:
                    actor = self._task_behavior.actor(feat)
                    action = actor.sample()
                noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
                action = common.action_noise(action, noise[mode], self._action_space)

                outputs = {'action': action.cpu()}  # no grads for env
                state = (latent, action)

                # no grads for env #FIXME but calling detach() multiple time no prob
                # outputs = {'action': action.detach().cpu()}
                # state = (common.dict_detach(latent), action.detach())
            return outputs, state

    def forward(self, data, state=None):
        return self.train(data, state)

    def train(self, data, state=None):
        metrics = {}

        # with torch.no_grad(): #FIXME pudia fazer flag
        state, outputs, mets = self.wm.train(data, state)  # outputs could propagate to behaviour

        metrics.update(mets)
        start = outputs['post']
        if self.config.pred_discount:  # Last step could be terminal.
            # start = tf.nest.map_structure(lambda x: x[:, :-1], start)
            start = {k: v[:, :-1].detach() for k, v in start.items()}
        else:
            start = common.dict_detach(start)  # detach post
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode
        metrics.update(self._task_behavior.train(self.wm, start, reward))
        # if self.config.expl_behavior != 'greedy':
        #     if self.config.pred_discount:
        #         # data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        #         # outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
        #         data = {k: v[:, :-1] for k, v in data.items()}  # FIXME check this
        #         outputs = {k: v[:, :-1] for k, v in outputs.items()}
        #     mets = self._expl_behavior.train(start, outputs, data)[-1] #FIXME outputs have previous graph from wm
        #     metrics.update({'expl_' + key: value for key, value in mets.items()})
        return common.dict_detach(state), metrics

    def report(self, data, state=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
                return {'openl': self.wm.video_pred(data, state).cpu().permute(0, 2, 3,
                                                                               1).numpy()}  # T,H,W,C needs to be numpy

    def init_optimizers(self):
        wm_modules = [self.wm.encoder.parameters(), self.wm.rssm.parameters(),
                      *[head.parameters() for head in self.wm.heads.values()]]
        self.wm.model_opt = common.Optimizer('model', wm_modules, **self.config.model_opt)

        self._task_behavior.actor_opt = common.Optimizer('actor', self._task_behavior.actor.parameters(),
                                                         **self.config.actor_opt)
        self._task_behavior.critic_opt = common.Optimizer('critic', self._task_behavior.critic.parameters(),
                                                          **self.config.critic_opt)


class WorldModel(common.Module):
    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self.step = step
        self.config = config
        self.rssm = common.RSSM(**config.rssm)
        self.heads = {}
        shape = (1 if config.grayscale else 3,) + config.image_size
        self.encoder = common.ConvEncoder(**config.encoder)

        self.heads = nn.ModuleDict({
            'image': common.ConvDecoder(shape, **config.decoder),
            'reward': common.MLP([], **config.reward_head),
        })
        if config.pred_discount:
            self.heads.update({
                'discount': common.MLP([], **config.discount_head)
            })
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.EmptyOptimizer()

    def train(self, data, state=None):
        with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
            self.zero_grad(set_to_none=True)  # delete grads

            model_loss, state, outputs, metrics = self.loss(data, state)

        # Backward passes under autocast are not recommended.
        self.model_opt.backward(model_loss)
        metrics.update(self.model_opt.step(model_loss))
        metrics['model_loss'] = model_loss.item()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data['action'], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else feat.detach()

            like = head(inp).log_prob(data[name])

            likes[name] = like
            losses[name] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value)  # stop propagating gradients? for now is okay, disabled in agent
        metrics = {f'{name}_loss': value.item() for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean().item()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean().item()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean().item()
        return model_loss, post, outs, metrics

    def imagine(self, policy, start, horizon):
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        start = {k: flatten(v) for k, v in start.items()}

        def step(state, _):
            # state, _, _ = prev
            feat = self.rssm.get_feat(state)

            action = policy(feat.detach()).sample()
            # action = policy(feat.detach()).mean  # for testing DEBUG
            with torch.no_grad():
                succ = self.rssm.img_step(state, action)

            return succ, feat, action

        # feat = 0 * self.rssm.get_feat(start)
        # action = policy(feat).mode  # NOt used?
        # succs, feats, actions = common.static_scan(
        #     step, tf.range(horizon), (start, feat, action))

        succs, feats, actions = common.sequence_scan(
            step, start, np.arange(horizon))
        states = {k: torch.cat([
            start[k][None], v[:-1]], 0) for k, v in succs.items()}

        with torch.no_grad():  # FIXME
            if 'discount' in self.heads:
                discount = self.heads['discount'](feats).mean
            else:
                discount = self.config.discount * torch.ones_like(feats[..., 0])

        return feats, states, actions, discount

    def preprocess(self, obs):
        # dtype = prec.global_policy().compute_dtype #everything is casted in data_loader next_batch
        obs = obs.copy()  # doesnt clone, but forces to clone on equals
        obs['image'] = obs['image'] / 255.0 - 0.5

        clip_function = lambda x: x if self.config.clip_rewards == 'identity' else getattr(torch,
                                                                                           self.config.clip_rewards)(x)
        obs['reward'] = clip_function(obs['reward'])
        if 'discount' in obs:
            obs['discount'] = obs['discount'] * self.config.discount
        return obs

    def video_pred(self, data, state=None):
        '''
        FIXME do transforms on cpu
        Log images reconstructions come from this function

        '''

        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5], state)
        recon = self.heads['image'](
            self.rssm.get_feat(states)).mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.rssm.get_feat(prior)).mode

        # select 6 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        # should do on cpu?
        recon = recon.cpu()
        openl = openl.cpu()
        truth = data['image'][:6].cpu() + 0.5

        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)  # time
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)  # on H
        B, T, C, H, W = video.shape  # batch, time, height,width, channels
        return video.permute(1, 2, 3, 0, 4).reshape(T, C, H, B * W)


#
class ActorCritic(common.Module):
    def __init__(self, config, step, num_actions):
        super(ActorCritic, self).__init__()
        self.config = config
        self.step = step
        self.num_actions = num_actions
        self.actor = common.MLP(num_actions, **config.actor)
        self.critic = common.MLP([], **config.critic)
        if config.slow_target:
            self._target_critic = common.MLP([], **config.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = common.EmptyOptimizer()
        self.critic_opt = common.EmptyOptimizer()

        self.once = 0

    def train(self, world_model, start, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
            # delete grads
            world_model.zero_grad(set_to_none=True)
            self.actor.zero_grad(set_to_none=True)
            self.critic.zero_grad(set_to_none=True)

            feat, state, action, disc = world_model.imagine(self.actor, start, hor)

            with torch.no_grad():  # FIXME only for mode reinforce is enabled
                reward = reward_fn(feat, state, action)

            target, weight, mets1 = self.target(feat, action, reward, disc)  # weight doesnt prop
            actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
            critic_loss, mets3 = self.critic_loss(feat, action, target, weight)

        # Backward passes under autocast are not recommended.
        self.actor_opt.backward(actor_loss, retain_graph=True)
        self.critic_opt.backward(critic_loss)

        metrics.update(self.actor_opt.step(actor_loss))
        metrics.update(self.critic_opt.step(critic_loss))
        metrics.update(**mets1, **mets2, **mets3)
        self.update_slow_target()
        return metrics

    def actor_loss(self, feat, action, target, weight):
        metrics = {}
        policy = self.actor(
            feat.detach())  # FIXME why would we do this again? use previous from imagine? deactivate grads?
        if self.config.actor_grad == 'dynamics':
            objective = target
        elif self.config.actor_grad == 'reinforce':
            baseline = self.critic(feat[:-1]).mode
            advantage = (target - baseline).detach()  # note here nothing props through critic
            objective = policy.log_prob(action)[:-1] * advantage  # note, no grads to action, only to policy
        elif self.config.actor_grad == 'both':
            baseline = self.critic(feat[:-1]).mode
            advantage = (target - baseline).detach()
            objective = policy.log_prob(action)[:-1] * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent[:-1]
        actor_loss = -(weight[:-1] * objective).mean()
        metrics['actor_ent'] = ent.mean().item()
        metrics['actor_ent_scale'] = ent_scale

        # debug
        mse_logits = (policy.orig_logits ** 2).mean()
        metrics['actor_logits_mse'] = mse_logits.item()

        metrics['z_actor_logits_policy_max'] = policy.orig_logits.max().item()  # unnormalized logits
        metrics['z_actor_logits_policy_min'] = policy.orig_logits.min().item()

        if torch.any(policy.logits.isnan()):
            print("actor logits nan")
        if torch.any(policy.logits.isinf()):
            print("actor logits inf")

        return actor_loss, metrics

    def critic_loss(self, feat, action, target, weight):
        # print("critic loss, ",feat.shape,target.shape,weight.shape)
        dist = self.critic(feat[:-1])
        target = target.detach()
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()  # log_prob of normal(out,1) is - mse
        metrics = {'critic': dist.mode.mean().item()}
        return critic_loss, metrics

    def target(self, feat, action, reward, disc):
        # print("target feat type",type(feat),"args:",feat.shape,reward.shape,disc.shape)
        # reward = tf.cast(reward, tf.float32) #FIXME verify casts
        # disc = tf.cast(disc, tf.float32)
        value = self._target_critic(feat).mode
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0)

        # weight -> discount**i, where weight[0] = discount**1
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0).detach(), 0)

        metrics = {}
        metrics['reward_mean'] = reward.mean().item()
        metrics['reward_std'] = reward.std().item()
        metrics['critic_slow'] = value.mean().item()
        metrics['critic_target'] = target.mean().item()
        metrics['discount'] = disc.mean().item()
        return target, weight, metrics

    def update_slow_target(self):  # polyak update
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
