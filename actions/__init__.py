"""
动作模块 - 自动注册所有动作类
"""
from .neutral_face import NeutralFaceAction
from .spontaneous_eye_blink import SpontaneousEyeBlinkAction
from .voluntary_eye_blink import VoluntaryEyeBlinkAction
from .close_eye_softly import CloseEyeSoftlyAction
from .close_eye_hardly import CloseEyeHardlyAction
from .raise_eyebrow import RaiseEyebrowAction
from .smile import SmileAction
from .shrug_nose import ShrugNoseAction
from .show_teeth import ShowTeethAction
from .blow_cheek import BlowCheekAction
from .lip_pucker import LipPuckerAction

# 动作注册表：动作名 -> 动作类
ACTION_REGISTRY = {
    'NeutralFace': NeutralFaceAction,
    'SpontaneousEyeBlink': SpontaneousEyeBlinkAction,
    'VoluntaryEyeBlink': VoluntaryEyeBlinkAction,
    'CloseEyeSoftly': CloseEyeSoftlyAction,
    'CloseEyeHardly': CloseEyeHardlyAction,
    'RaiseEyebrow': RaiseEyebrowAction,
    'Smile': SmileAction,
    'ShrugNose': ShrugNoseAction,
    'ShowTeeth': ShowTeethAction,
    'BlowCheek': BlowCheekAction,
    'LipPucker': LipPuckerAction
}


def get_action_class(action_name):
    """
    获取动作类

    Args:
        action_name: 动作名称（英文）

    Returns:
        动作类
    """
    return ACTION_REGISTRY.get(action_name)


__all__ = [
    'ACTION_REGISTRY',
    'get_action_class',
    'NeutralFaceAction',
    'SpontaneousEyeBlinkAction',
    'VoluntaryEyeBlinkAction',
    'CloseEyeSoftlyAction',
    'CloseEyeHardlyAction',
    'RaiseEyebrowAction',
    'SmileAction',
    'ShrugNoseAction',
    'ShowTeethAction',
    'BlowCheekAction',
    'LipPuckerAction'
]