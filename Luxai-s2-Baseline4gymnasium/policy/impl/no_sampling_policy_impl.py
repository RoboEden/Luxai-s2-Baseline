from plf.base import registry
from plf.policy.impl.policy_impl import _PolicyImpl


@registry.register("no_sampling_policy")
class _NoSamplingPolicyImpl(_PolicyImpl):

    def sample(self, input_dict):
        o = self.infer(input_dict)
        return o

    def batch_sample(self, input_dict_list):
        output_dict_list = self.batch_infer(input_dict_list)
        return output_dict_list
