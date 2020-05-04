# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function


from .cmplx import fComplex, fParamComplex
from .arrays import fExplicitArray, fDummyArray, fParamArray, fAssumedShape, fAssumedSize
from .strings import fStr
from .var import fVar, fParam


def _selectVar(obj):
    x = None

    if 'param' in obj:
        # Parameter strings are just be fParams
        if obj['param']['pytype'] == 'complex':
            x = fParamComplex
        elif obj['param']['array']:
            if obj['param']['array']:
                x = fParamArray
        else:
            x = fParam
    elif 'var' in obj:
        if obj['var']['pytype'] == 'str':
            x = fStr
        elif obj['var']['pytype'] == 'complex':
            x = fComplex
        elif 'dt' in obj['var'] and obj['var']['dt']:
            from .types import fDerivedType
            x = fDerivedType
        elif 'array' in obj['var']:
            array = obj['var']['array']['atype']
            if array == 'explicit':
                x = fExplicitArray
            elif array == 'assumed_shape' or array == 'alloc' or array == 'pointer':
                x = fAssumedShape
            elif array == 'assumed_size':
                x = fAssumedSize
        elif 'is_func' in obj['var'] and obj['var']['is_func']:
            from .functions import fFuncPtr
            x = fFuncPtr
        else:
            x = fVar

    return x
