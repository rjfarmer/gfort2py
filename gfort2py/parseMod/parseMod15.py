# SPDX-License-Identifier: GPL-2.0+

from . import parseModCommon as pmc
from .utils import *
from .utils_cpython import *
import ctypes

class parseMod(pmc.parseModBase):
    def __init__(self,*args):
        self.mod_version=14
        super(parseMod,self).__init__(*args)

    def getParamValue(self,x,typ):
        if 'ARRAY' in x:
            x = split_brackets(x.strip())
            arrLen = int(''.join(c for c in x[-2] if c not in "'()")) # Changed in mod15
            x2 = split_brackets(x[1][x[1].index("("):])
            listParam = [y.split("'")[-2] for y in x2]
            p = [self.parseSingleParam(y,typ) for y in listParam]
        else:
            if 'COMPLEX' in x:
                yy = x.split("'")
                p = complex(self.parseSingleParam(yy[-4],'float'),
                     self.parseSingleParam(yy[-2],'float'))
            else:
                y = x.split()[-2] # Changed in mod15
                y = y.replace("'","")
                p = self.parseSingleParam(y,typ) 
            
        return p

    def parseDT(self,info):
        res={}
        res['arg']=[]
        
        e = split_brackets(info[1].strip())
        for i in e:
            #Remove whitespace and the first and last bracket
            i=i.strip()[1:-1]
            dtEl={}
            dtEl['num'], dtEl['name'] = i.split()[0:2]
            dtEl['name'] = dtEl['name'].lower().replace("'","")
            info_el = split_brackets(i[i.index("(")-1:],remove_b=False)
            #Re-order to be the same as everything else
            newL = [info_el[-2],'()',info_el[0],'()',info_el[1]]
            dtEl['var'] = self.parseVar(newL)
            #Fix size 
            if 'dt' in dtEl['var']:
                dtEl['var']['bytes'] = ctypes.sizeof(ctypes.c_void_p)
            res['arg'].append(dtEl)

        return res
