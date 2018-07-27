# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
try:
    import __builtin__ 
except ImportError:
    import builtins as __builtin__
import gzip
import ctypes
import os
import pickle
import sys
import re
import subprocess
import numpy as np
import multiprocessing as mp

from .utils import *
from .utils_cpython import *

PARALLEL = False

class parseModBase(object):
    def __init__(self,data,filename,res,PYFILE_VERSION):
        self.data = data
        self.filename = filename
        self.mod_data = res
        self.PYFILE_VERSION = PYFILE_VERSION
        
        self.funcs={}
        self.mod_vars={}
        self.param={}
        self.dt_defs={}
        self.dt_defines=[]
        self._unpacked=False
    
    def processData(self):
        self.data = split_brackets(self.data)
        
        #Store the split data once:
        _t=[]
        for i in self.data:
            _t.append(split_brackets(i.strip(),remove_b=True))
        self.data = _t
        
    
        self.parseAllIntrinsic()
        self.parseAllOperator()
        self.parseAllDTDefines()
        self.parseAllCommon()
        self.parseAllEqv()
        self.parseAllNameSymbols()
        self.parseAllSymbols()
        
        self.matchFuncArgs()
        self.data=None
        
        self.unpackData()
    
    def save(self,output):
        if not self._unpacked:
            self.unpackData()
        
        self.pickler(output, self.PYFILE_VERSION, 
                    self.mod_data, self.mod_vars, self.param, 
                    self.funcs, self.dt_defs)
    
    
    def pickler(self,filename, *args):   
        with open(filename, 'wb') as f:
            for i in args:
                pickle.dump(i, f, protocol=2)
            
    def unpackData(self):
        self._unpacked = True
        
        for i in self.all_symbols:
            if len(i):
                try:
                    name = i['name']
                except KeyError:
                    print(i)
                    name = i['mangled_name']
                if 'proc' in i:
                    self.funcs[name] = i
                elif 'var' in i and 'func_arg' not in i:
                    self.mod_vars[name] = i
                elif 'param' in i:
                    self.param[name] = i
                elif 'dt_def' in i:
                    self.dt_defs[name] = i
                
    def getUnpackedData(self):
        if not self._unpacked:
            self.unpackData()
            
        return  self.mod_data, self.mod_vars, self.param, self.funcs, self.dt_defs
    
    def mangleName(self,item):
        return '__' + item['module'] + '_MOD_' + item['name'].lower()
        
    def parseAllIntrinsic(self):
        x = self.data[0]
        #Remove the module header
        #x = x[x.index("("):]
        
    def parseAllOperator(self):
        x = self.data[1]
        
    def parseAllDTDefines(self):
        # This isnt just dt names, its also the name and number(s) for 
        # generic interfaces
        x = self.data[2]
        #x = split_brackets(x.strip(),remove_b=True) 
        
        for i in x:
            i=i.replace("(","").replace(")",'').strip()
            r={}
            z = i.split()
            r['name'] = z[0].lower().replace("'","")
            r['module'] = z[1].replace("'","")
            r['num'] = int(z[2])
            if(len(z))> 3:
                r['num_alts']=z[2:]
            self.dt_defines.append(r)
            
            
        # Find the dt's used by the module but not defined by the module,
        # but its defintion is still copied in here
        for symbol in self.data[6]:
            try:
                splitPoint = symbol.index("(")
            except ValueError:
                continue
            info = symbol[splitPoint:].strip()
            info = split_brackets(info)
            type_line = info[0]
            
            if 'DERIVED ' in type_line:
                num,name,module  =  symbol.split()[0:3]
                if len(self.dt_defines)>0:
                    found=False
                    for i in self.dt_defines:
                        if int(i['num']) == int(num):
                            found=True
                else:
                    found = False
                
                if not found:
                    self.dt_defines.append({'name':name,
                                        'module':module.replace("'",""),
                                        'num':num})
    
    def parseAllCommon(self):
        x = self.data[3]
        
    def parseAllEqv(self):
        x = self.data[4]
    
    def parseAllNameSymbols(self):
        x = self.data[7]
        x = x[0].split()
        self.symbol_names = []
        for i in range(len(x)//3):
            self.symbol_names.append({'name':x[i*3],'num':x[i*3+2],'ambiguous_flag':bool(x[i*3+1])})
        
    
    def parseAllSymbols(self):
        split_data = self.data[6]
        
        if PARALLEL:
            with mp.Pool() as pool:
                all_symbols = pool.map(self.parseSymbol,split_data)
        else:
            all_symbols = [self.parseSymbol(i) for i in split_data]
            
        self.all_symbols = all_symbols
        
    def parseSymbol(self,symbol):
        symbol = symbol.strip()
        #things to skip
    
        if ('__' in symbol or '(intrinsic)' in symbol or 
            'INTRINSIC' in symbol or len(symbol)==0 or
            ' RESULT' in symbol or 'BODY' in symbol):
            return {}
    
        r = {}
        r['num'], r['name'],r['module'], _, r['parent_id']  =  symbol.split()[0:5]
        
        for i in ['num','name','module','parent_id']:
            r[i]=r[i].lower().replace("'","")
        
        if len(r['name'])==0:
            return {}
        
        ## things to left of '(' are basic info right of '(' contains detailed info
        ## about the symbol  
        splitPoint = symbol.index("(")
        
        info = symbol[splitPoint:].strip()
        
        
        info = split_brackets(info)
        r['info'] = info
        type_line = info[0]
        # need spaces on end of names as sometimes we have name-something etc
        if 'VARIABLE ' in type_line or 'DUMMY ' in type_line:
            if len(r['module'])>0:
                r['var'] = self.parseVar(info)
            else:
                r['var'] = self.parseFuncArg(info)
                r['func_arg']=True
        elif 'PROCEDURE ' in type_line:
            if 'GENERIC' in type_line:
                pass #Uneeded
                #r['dt_type'] = self.parseDTType(info)
            else:
                r['proc'] = self.parseProc(info)
                r['arg']=[]
        elif 'MODULE ' in type_line:
            r['module_info'] = self.parseModule(info)
        elif 'PARAMETER ' in type_line:
            r['param'] = self.parseParam(info)
        elif 'DERIVED ' in type_line:
            #This definition of a dt 
            r['dt_def'] = self.parseDT(info)
        elif 'NAMELIST ' in type_line:
            print("Skipping namelist "+r['name'])
        else:
            raise ValueError("Unknown object "+symbol)
            
        r['mangled_name']=self.mangleName(r)
            
        r.pop('info')
    
        return r 
        
       
    def parseVar(self,info):
        res={}
    
        type_info = info[2]
        symbol_info = info[0]
        p, c, s=self.getVarType(type_info)
        
        res['pytype'] = p
        res['ctype'] = c
        res['bytes'] = s
        
        if res['pytype'] == 'str':
            res['length'] = self.getStrLen(info)
        
        if 'DIMENSION' in symbol_info:
            #Array
            res['array'] = self.processArray(info)
        if 'DERIVED' in type_info:
            #Both dts but sometimes the definition moves
            res['dt'] = self.processDTVar(info)
            
        res['optional']=False
        if 'POINTER' in symbol_info:
            res['pointer']=True
        if 'TARGET' in symbol_info:
            res['target']=True
        if 'OPTIONAL' in symbol_info:
            res['optional']=True
        if 'DUMMY' in symbol_info:
            res['intent'] = self.parseIntent(symbol_info)
        if 'DUMMY-PROC' in symbol_info:
            res['is_func'] = True
    
        return res
        
    def parseProc(self,info):
        res={}
        type_info = info[2]
        
        if 'FUNCTION ' in info[0]:        
            p, c, s=self.getVarType(type_info)
        else:
            p = None
            c = None
            s = None
            
        res['pytype'] = p
        res['ctype'] = c
        res['bytes'] = s
        
        x=info[3][info[3].index("("):]
        res['arg_nums']=x.replace("(","").replace(")","").split()
        
        if 'SUBROUTINE' in info[0]:
            res['sub']=True
        else:
            res['sub']=False
        
        #Return value
        res['ret'] = self.parseVar(info)
        
        func_args = info[3].split()
        res['func_arg_id'] = int(func_args[0])
        
        if 'ABSTRACT 'in type_info:
            res={}
        
        
        return res
        
    def parseModule(self,info):
        res={}
        type_info = info[2]
        return res
        
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
            newL = [info_el[2],'()',info_el[0],'()',info_el[1]]
            dtEl['var'] = self.parseVar(newL)
            #Fix size 
            if 'dt' in dtEl['var']:
                dtEl['var']['bytes'] = ctypes.sizeof(ctypes.c_void_p)
            res['arg'].append(dtEl)
            
        
        return res
        
    def parseFuncArg(self,info):
        r=self.parseVar(info)
        return r
        
    def parseParam(self,info):
        res={}
        
        type_info = info[2]
        value = info[4]
        
        p, c, s=self.getVarType(type_info)
        
        res['pytype'] = p
        res['ctype'] = c
        res['bytes'] = s
        
        res['value']=self.getParamValue(value,res['pytype'])
        
        res['array']=False
        if 'ARRAY' in value:
            res['array']=True
        
        return res
    
    
    def getVarType(self,x):
        x=x.strip()
        if x.startswith('('):
            x=x[1:].strip()
       
        x = x.split()
        size = x[1]
        pytype, ctype = self.getTypes(x[0],size)
        
        return pytype, ctype, size
        
    def getParamValue(self,x,typ):
        if 'ARRAY' in x:
            x = split_brackets(x.strip())
            arrLen = int(''.join(c for c in x[-1] if c not in "'()"))
            x2 = split_brackets(x[1][x[1].index("("):])
            listParam = [y.split("'")[-2] for y in x2]
            p = [self.parseSingleParam(y,typ) for y in listParam]
        else:
            if 'COMPLEX' in x:
                yy = x.split("'")
                p = complex(self.parseSingleParam(yy[-4],'float'),
                     self.parseSingleParam(yy[-2],'float'))
            else:
                y = x.split()[-1][:-1]
                y = y.replace("'","")
                p = self.parseSingleParam(y,typ) 
            
        return p
    
    def parseSingleParam(self,x,typ):
        if '@' in x:
            return hextofloat(x)
        else:
            return getattr(__builtin__, typ)(x)
            
    def processArray(self,info):
        r = {}
        if 'ALLOCATABLE' in info[0]:
            r['atype'] = 'alloc'
        elif 'POINTER' in info[0]:
            r['atype'] = 'pointer'
        elif 'ASSUMED_SHAPE' in info[4]:
            r['atype'] = 'assumed_shape'
        elif 'ASSUMED_SIZE' in info[4]:
            r['atype'] = 'assumed_size'
        elif 'CONSTANT' in info[4]:
            r['shape'] = self.getBounds(info)
            r['atype'] = 'explicit'
        r['ndim'] = self.getNdims(info)
        
        p, c, s=self.getVarType(info[2])
        
        r['pytype'] = p
        r['ctype'] = c
        r['bytes'] = s
        
        return r
        
    def parseDTType(self, info):
        #Nothing of any use in the info list
        return None
        
    def getNdims(self, info):
        return info[4].replace("(", "").strip().split()[0]
    
    def getBounds(self, info):
        try:
            # Horrible but easier than splitting the nested brackets
            return [int(x) for x in info[4].split("'")[1:-1:2]]
        except ValueError:
            # Sometimes we cant know the size till run time
            return -1
    
    def getStrLen(self, info):
        # Horrible but easier than splitting the nested brackets
        y=info[2].split("'")[1:-1:2]
        if len(y)==0:
            y=-1
        else:
            y=y[0]
        return int(y)
    
    def processDTVar(self,info):
        # size is actually the dt definition number
        p, c, s=self.getVarType(info[2])
        return self.matchDtDef(int(s))
    
    
    def getTypes(self,x,size):
        if 'CHARACTER' in x:
            pytype='str'
            ctype='c_char_p'
        elif 'INTEGER' in x:
            pytype='int'
            ctype=self.getCtypeIntSize(size)
        elif 'REAL' in x:
            pytype,ctype=self.getCtypeFloatSize(size)
        elif 'COMPLEX' in x:
            pytype,ctype=self.getCtypeFloatSize(size)
            pytype='complex'
        elif 'LOGICAL' in x:
            pytype='bool'
            ctype='c_int'
        elif 'UNKNOWN' in x:
            pytype='None'
            ctype='c_void_p'
        elif "DERIVED" in x:
            pytype='dict'
            ctype='c_void_p'
        else:
            pytype='None'
            ctype='c_void_p'
            #raise ValueError("Cant parse " + x)
            
        if 'PASS ' in x:
            pytype='None'
            ctype='c_void_p'
            
        return pytype,ctype
        
    def getCtypeIntSize(self,size):
        size = int(size)
        if size == ctypes.sizeof(ctypes.c_int):
            res = 'c_int'
        elif size == ctypes.sizeof(ctypes.c_int16):
            res = 'c_int16'
        elif size == ctypes.sizeof(ctypes.c_int32):
            res = 'c_int32'
        elif size == ctypes.sizeof(ctypes.c_int64):
            res = 'c_int64'
        elif size == ctypes.sizeof(ctypes.c_byte):
            res = 'c_byte'
        elif size == ctypes.sizeof(ctypes.c_short):
            res = 'c_short'
        else:
            raise ValueError("Cant find suitable int for size " + str(size))
        return res
    
    def getCtypeFloatSize(self,size):
        size = int(size)
        pytype='float'
        if size == ctypes.sizeof(ctypes.c_float):
            res = 'c_float'
        elif size == ctypes.sizeof(ctypes.c_double):
            res = 'c_double'
        elif size == ctypes.sizeof(ctypes.c_long):
            res = 'c_long'
        elif size == ctypes.sizeof(ctypes.c_longdouble):
            res = 'c_longdouble'
            pytype='quad'
        elif size == ctypes.sizeof(ctypes.c_longlong):
            res = 'c_longdouble'
            pytype='quad'
        else:
            raise ValueError("Cant find suitable float for size " + str(size))
        return pytype,res
    
    def parseIntent(self,info):
        value = False
        if ' INOUT ' in info:
            value = 'inout'
        elif ' OUT ' in info:
            value = 'out'
        elif ' IN ' in info:
            value = 'in'
        elif ' UNKNOWN-INTENT ' in info:
            value = 'na'
        return value
    
    def matchDtDef(self,num):
        for i in self.dt_defines:
            if int(num) == int(i['num']):
                return i
        
        print("Cant match dt definition "+str(num))
    
    def matchFuncArgs(self):
        ind_funcs=[]
        ind_func_args=[]
        
        func_nums=[]
        func_arg_num=[]
        
        for idx,i in enumerate(self.all_symbols):
            if 'proc' in i:
                ind_funcs.append(idx)
                func_nums.append(int(i['proc']['func_arg_id']))
            if 'func_arg' in i:
                ind_func_args.append(idx)
                func_arg_num.append(int(i['parent_id']))
    
        s_f = np.argsort(func_nums,kind='heapsort')
    
        sorted_index = np.searchsorted(func_nums,func_arg_num,sorter=s_f,side='left')
                
        lenfn=len(s_f)
        # Things with index greater than the length of the function nums are
        # arguments to functions that are argument to funcs
        sorted_index = [i if i<lenfn else None for i in sorted_index]
    
        for i,j in zip(sorted_index,ind_func_args):
            if i is not None:
                fAind=s_f[i]
                self.all_symbols[ind_funcs[fAind]]['arg'].append(self.all_symbols[j])
        
    
    
    
    
