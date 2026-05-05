import sys
from ._base_steer import Steer

# Vector-based Steers
from ._repe import RepE
from ._caa import CAA
from ._iti import ITI

# OT-based Steers
from ._mimic import MiMiC
from ._lin_act import LinAcT

# CRFDSteer (legacy aliases: ODESteer / StepODESteer / RFF*)
from ._ode_steer import (
    BaseCRFDSteer,
    CRFDSteer,
    RFFCRFDSteer,
    BaseODESteer,
    ODESteer,
    RFFODESteer,
)
from ._step_ode_steer import (
    BaseStepCRFDSteer,
    StepCRFDSteer,
    RFFStepCRFDSteer,
    BaseStepODESteer,
    StepODESteer,
    RFFStepODESteer,
)

__all__ = [
    'Steer', 'VecSteer', 
    # Baselines
    'RepE', 'CAA', 'ITI', 'MiMiC', 'LinAcT',
    # CRFDSteer (primary)
    'BaseCRFDSteer', 'CRFDSteer', 'RFFCRFDSteer',
    'BaseStepCRFDSteer', 'StepCRFDSteer', 'RFFStepCRFDSteer',
    # Legacy (ODESteer names)
    'BaseODESteer', 'ODESteer', 'RFFODESteer',
    'BaseStepODESteer', 'StepODESteer', 'RFFStepODESteer',
]

def get_steer_model(name: str, *args, **kwargs) -> type[Steer]:
    if name == "NoSteer":
        return None
    return getattr(sys.modules[__name__], name)(*args, **kwargs)
