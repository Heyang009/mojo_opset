"""UC wrapper for apply-penalties temperature sampling.

The current uc-kernel ABI does not cover the full operator without torch-side
tensor preparation.
"""

from mojo_opset.core import MojoApplyPenaltiesTempurate


class UCApplyPenaltiesTempurate(MojoApplyPenaltiesTempurate):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UCApplyPenaltiesTempurate is not implemented as a single direct uc-kernel call."
        )
