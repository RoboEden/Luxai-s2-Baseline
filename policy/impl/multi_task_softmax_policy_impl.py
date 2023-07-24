import math
from dataclasses import asdict
from functools import partial

import numpy as np
import scipy
import tree
from impl_config import FullAct, ModelParam, UnitActType
from plf.base import registry
from plf.policy.impl.policy_impl import _PolicyImpl


def _gen_pi(va, logits):  # task_size * act_dim
    ceil = math.pow(10.0, 20)
    # pi maybe problematic when action number > 100000
    # if logits.shape[0] >= 100000:
    #     glog.warn("Warning: action number greater than 100000 may case a problem.")

    # softmax, masked invalid action with considerably large ne

    logits = np.where(va, logits, np.full_like(logits, -ceil))
    logits_max = np.amax(logits, axis=0, keepdims=True)

    # valid logits may lower than -ceil
    logits_exp = va * np.exp(np.clip(logits - logits_max, -ceil, 1))
    sum_logits_exp = np.sum(logits_exp, axis=0, keepdims=True) + np.finfo(np.float32).tiny
    pi = logits_exp / sum_logits_exp
    return pi


def _sample_til_valid(pi: np.ndarray, va: np.ndarray):
    dim, H, W = pi.shape
    if np.sum(va) == 0:
        action = np.zeros((H, W), dtype=np.int32)  # random action,
        return action

    rand = np.random.random(size=(H, W))
    right = np.cumsum(pi, axis=0)
    left = np.roll(right, 1, axis=0)
    left[0] = 0
    action = np.argmax(((rand >= left) & (rand < right)), axis=0)

    sample_va = va[action, np.arange(va.shape[1])[:, None], np.arange(va.shape[2])]
    is_valid = sample_va | (np.sum(va, axis=0) == 0)
    invalid_pos = np.where(~is_valid)
    assert is_valid.all(), \
        f"invalid action at {invalid_pos}, with \n"\
        f"va={va[(slice(None),)+ invalid_pos]}\n"\
        f"pi={pi[(slice(None),)+ invalid_pos]}\n"\
        f"random={rand[invalid_pos]}\n"\
        f"cumpi={right[(slice(None),)+ invalid_pos]}\n"\
        f"action={action[invalid_pos]}\n"
    return action


def _logp(va, logits, act):
    logits = np.where(va, logits, -1e20)
    logp = logits - scipy.special.logsumexp(logits, axis=0, keepdims=True)

    logp = logp[
        act, \
        np.arange(act.shape[0]).reshape(-1, 1), \
        np.arange(act.shape[1]).reshape(1, -1), \
    ]

    is_valid = ~np.isinf(logp) & ~np.isnan(logp)
    invalid_pos = np.where(~is_valid)
    assert is_valid.all(), \
        f"invalid action at {invalid_pos}, with \n"\
        f"logits={logits[(slice(None),)+ invalid_pos]}\n"\
        f"act={act[invalid_pos]}\n"\
        f"logp={logp[invalid_pos]}\n"

    return logp


def lux_tree_softmax(o):  # mapping:{0:{},1:{},2:{}}
    for agent in o["agents"]:
        agent["pi"] = tree.map_structure(_gen_pi, agent["va"], agent["logits"])
        agent["action"] = tree.map_structure(_sample_til_valid, agent["pi"], agent["va"])
        agent["logp"] = tree.map_structure(_logp, agent["va"], agent["logits"], agent["action"])
    return o


@registry.register("lux_softmax_policy")
class _SoftmaxPolicyImpl(_PolicyImpl):
    r"""
    Policy selects actions based on softmax probability.
    """

    def __init__(
        self,
        # mapping=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._softmax = partial(lux_tree_softmax)

    def sample(self, input_dict):
        r"""
        Samples actions from policy with valid actions.

        Output of :class:`~plf.policy.model.model.Model.forward` should 
        have the following structure::

            output = {
                "agents": 
                [
                    {
                        "logits": tensorflow.Operation,
                    },
                    ...,
                ]
            }
        
        input_dict should have following structure::
        
            input_dict = {
                "agents":
                [
                    {
                        "obs": any type as input
                        "va": numpy.Array (optional, if not given, valid action will be all actions avaliable)
                    }
                ]
            }

        Args:
            input_dict (dict): A dict of inputs.

        Returns:
            A dict with structure::

                output = {
                    "agents": 
                    [
                        {
                            "pi": numpy.Array,
                            "action": int,
                            "logp": float,
                            "value": float, (if exist)
                        },
                        ...,
                    ]
                }
        """
        o = self.infer(input_dict)
        o = self._softmax(o)

        return o

    def batch_sample(self, input_dict_list):
        output_dict_list = self.batch_infer(input_dict_list)
        return [self._softmax(output_dict) for output_dict in output_dict_list]