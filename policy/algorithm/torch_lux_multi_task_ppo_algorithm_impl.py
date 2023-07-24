import torch
from plf.base import registry, torch_helper
from plf.policy.algorithm.impl.algorithm_impl import _AlgorithmImpl
import tree


def _rn(key, i):
    return "a{}_{}".format(i, key)


@registry.register("torch_lux_multi_task_ppo_algorithm")
class _TorchPPOAlgorithm(_AlgorithmImpl):
    r"""
    Implementation of ``PPO`` algorithm.

    Args:
        clip_ratio (float): [description]
        value_coef (float, optional): [description]. Defaults to 1.0.
        entropy_coef (float, optional): [description]. Defaults to 0.0.
        adv_norm (bool, optional): [description]. Defaults to True.
    """

    def __init__(
        self,
        clip_ratio,
        value_coef=1.0,
        entropy_coef=0.0,
        do_value_clip=True,
        do_adv_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.clip_ratio = float(clip_ratio)
        self.do_value_clip = bool(do_value_clip)
        self.do_adv_norm = bool(do_adv_norm)

    def loss(self, input_dict):
        s_loss = None
        o = {}

        for i, p in enumerate(input_dict["agents"]):
            ret = p["return"]  # shape: batch_size x 1
            value = p["value"]  # shape: batch_size x 1
            value_old = p["value_old"]  # shape: batch_size x 1
            is_train = p["is_train"]  # shape batch_size

            devi = ret.device

            policy_loss = torch.tensor(0, dtype=torch.float32, device=devi)
            entropy_loss = torch.tensor(0, dtype=torch.float32, device=devi)
            advantage = torch.tensor(0, dtype=torch.float32, device=devi)
            approx_kl = torch.tensor(0, dtype=torch.float32, device=devi)
            policy_clip_frac = torch.tensor(0, dtype=torch.float32, device=devi)
            value_clip_frac = torch.tensor(0, dtype=torch.float32, device=devi)
            valid_frac = torch.tensor(0, dtype=torch.float32, device=devi)
            # value loss
            value_diff = value - value_old
            clip_value = value_old + torch.clamp(value_diff, -self.clip_ratio, self.clip_ratio)

            if self.do_value_clip:
                value_loss = 0.5 * ((torch.max((value - ret)**2, (clip_value - ret)**2)).mean())
            else:
                value_loss = 0.5 * ((value - ret)**2).mean()

            adv = ret - value_old
            if self.do_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            if True:
                logp = p["logp"]
                logp_old = p["logp_old"]
                entropy = p["entropy"]

                #ratio = torch.exp(logp - logp_old)
                # (bs,) -> (bs, 1)
                sub = (logp - logp_old) * is_train  # batchsize*n_task

                ratio = torch.exp(sub)
                surr1 = torch.clamp(ratio, 0.0, 10000) * adv
                surr2 = torch.clamp(ratio, 1. - self.clip_ratio, 1. + self.clip_ratio) * adv

                # ref: https://discuss.pytorch.org/t/max-of-a-tensor-and-a-scalar/1436/2
                # (bs, 1) -> (bs,)
                policy_loss -= torch.sum(torch.min(surr1, surr2) * is_train) / torch.clamp(torch.sum(is_train), min=1.0)

                # entropy loss
                entropy_loss -= torch.sum(entropy * is_train) / torch.clamp(torch.sum(is_train), min=1.0)

                # additional info
                advantage += torch.sum(adv * is_train) / torch.clamp(torch.sum(is_train), min=1.0)

                approx_kl += (logp_old - logp).pow(2).mean() * 0.5
                policy_clip_frac += torch.as_tensor(ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio),
                                                    dtype=torch.float32).mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            if s_loss is None:
                s_loss = torch.zeros_like(loss)

            s_loss += loss
            value_clip_frac += torch.as_tensor(
                value_diff.gt(self.clip_ratio) | value_diff.lt(-self.clip_ratio),
                dtype=torch.float32,
            ).mean()

            o.update({
                _rn("total_loss", i): loss,
                _rn("value_loss", i): value_loss,
                _rn("policy_loss", i): policy_loss,
                _rn("value_mean", i): value.mean(),
                _rn("return_mean", i): ret.mean(),
                _rn("entropy_loss", i): entropy_loss,
                _rn("approx_kl", i): approx_kl,
                _rn("value_diff", i): value_diff.mean(),
                _rn("policy_clip_frac", i): policy_clip_frac,
                _rn("value_clip_frac", i): value_clip_frac,
            })

        o["loss"] = s_loss
        return o
