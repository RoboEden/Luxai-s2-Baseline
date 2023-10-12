import torch
from torch import nn

from policy.net import Net
import tree


@registry.register("torch_lux_model")
class _TorchPlayerModelImpl(_TorchModelImpl):

    def __init__(self, init_from=None, random_init_critic=True, **kwargs):
        self.init_from = init_from
        self.random_init_critic = random_init_critic
        super().__init__(**kwargs)

    def _build_net(self):
        self._net = Net().to(torch_helper.device)
        if self.init_from:
            state_dict = torch.load(self.init_from, map_location=torch_helper.device)
            for name, param in self.named_parameters():
                if name in state_dict:
                    if state_dict[name].shape != param.shape \
                        or state_dict[name].dtype != param.dtype:
                        state_dict.pop(name)
            self._net.load_state_dict(state_dict, strict=False)
            if self.random_init_critic:
                self._net._init_critic_net()
        return self._net

    def forward(self, input_dict):
        p = input_dict["agents"][0]
        training = input_dict.get("_train", None)
        p["logp"], p["value"], p['action'], p['entropy'] = self._net.forward(
            p["global_feature"],
            p["map_feature"],
            p["action_feature"],
            tree.map_structure(lambda x: x.bool(), p['va']),
            training and p['action'],
        )

        return input_dict
