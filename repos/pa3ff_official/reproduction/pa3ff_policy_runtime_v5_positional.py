"""Inference runtime for positional, task-balanced PADP V5."""
from pa3ff_policy_runtime_v4_baseframe import PA3FFPADPRuntimeV4BaseFrame


class PA3FFPADPRuntimeV5Positional(PA3FFPADPRuntimeV4BaseFrame):
    CHECKPOINT_STAGE = "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
    POSITIONAL_POINT_TOKENS = True


PA3FFPADPRuntimeManyCandidatesV5 = PA3FFPADPRuntimeV5Positional
