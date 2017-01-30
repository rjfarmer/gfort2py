import gzip
import ctypes
import os
import pickle
import sys
import re
			
def clean_list(l,idx):
	return [i for j, i in enumerate(l) if j not in idx]
	
def mangle_name(obj):
	return '__'+obj['module_name']+'_MOD_'+obj['name']
	

def split_brackets(value,remove_b=False):
	'''
	Split a string based on pairs of brackets, nested brackest are not split
	
	Input:
		'abc (def) (fgh () ())'
	
	Outputs:
		['def', 'fgh () ()']
	'''
	res=[]
	token=''
	start=False
	count=0
	for i in value:
		if i=='(':
			count=count+1
			start=True
		if start:
			token+=i
		if i==')':
			count=count-1
		if start and count==0:
			if remove_b:
				res.append(token[1:-1])
			else:
				res.append(token)
			start=False
			token=''
	return res

def find_key_val(list_dicts,key,value):
	for idx,i in enumerate(list_dicts):
		if i[key]==value:
			print(idx)
			print(i)	

def load_data(filename):
	try:
		with gzip.open(filename) as f:
			x=f.read()
	except OSError as e:
		e.args=[filename+" is not a valid .mod file"]
		raise
	x=x.decode()
	mod_data=get_mod_data(x)
	x=x.replace('\n',' ')
	data=split_brackets(x)
	return data,mod_data
	
def get_mod_data(x):
	header=x.split('\n')[0]
	res={}
	if 'GFORTRAN' not in header:
		raise AttributeError('Not a gfortran mod file')
	res['version']=int(header.split("'")[1])
	res['orig_file']=header.split()[-1]
	
	if not res['version']==14:
		raise AttributeError('Unsupported mod file version')
	
	return res

def split_info(info):
	#Cleanup brackets
	if info[0:2]=="((":
		info=info[1:]
	#if info[-2:]==' )':
		##extra bracket on final element in the orignal data
		#info=info[:-1]
	sp=split_brackets(info)
	return sp
	
def get_ctype_int(size):
	size=int(size)
	if size==ctypes.sizeof(ctypes.c_int):
		res='c_int'
	elif size==ctypes.sizeof(ctypes.c_int16):
		res='c_int16'
	elif size==ctypes.sizeof(ctypes.c_int32):
		res='c_int32'
	elif size==ctypes.sizeof(ctypes.c_int64):
		res='c_int64'
	elif size==ctypes.sizeof(ctypes.c_byte):
		res='c_byte'
	elif size==ctypes.sizeof(ctypes.c_short):
		res='c_short'		
	else:
		raise ValueError("Cant find suitable int for size "+str(size))	
	return res
	
def get_ctype_float(size):
	size=int(size)
	if size==ctypes.sizeof(ctypes.c_float):
		res='c_float'
	elif size==ctypes.sizeof(ctypes.c_double):
		res='c_double'
	elif size==ctypes.sizeof(ctypes.c_long):
		res='c_long'
	elif size==ctypes.sizeof(ctypes.c_longdouble):
		res='c_longdouble'
	elif size==ctypes.sizeof(ctypes.c_longlong):
		res='c_long'
	else:
		raise ValueError("Cant find suitable float for size "+str(size))
	return res


def parse_type(info,dt=False):
	if dt:
		#Derived types have own format, must remove ( and space at start
		attr=info[0][1:]
		size=attr.strip().split()[1]
	else:	
		attr=info[2]
		size=attr.split()[1]
	if 'INTEGER' in attr:
		pytype='int'
		ctype=get_ctype_int(size)
	elif 'REAL' in attr:
		pytype='float'
		ctype=get_ctype_float(size)
	elif 'COMPLEX' in attr:
		pytype='void'
		ctype=get_ctype_float(size)
	elif 'CHARACTER' in attr:
		pytype='str'
		ctype='c_char'
	elif 'LOGICAL' in attr:
		pytype='bool'
		ctype='c_bool'
	elif 'DERIVED' in attr:
		pytype='void'
		ctype='void'
	elif 'UNKNOWN' in attr:
		pytype='void'
		ctype='void'			
	else:
		raise ValueError("Cant parse "+attr)
	return pytype,ctype
	
def parse_array(info,dt=False):
	d={}
	if dt:
		attr1=info[2]
		attr2=info[1]
	else:
		attr1=info[0]
		attr2=info[4]
	
	if not 'DIMENSION' in attr1:
		return False
	if 'ALLOCATABLE' in attr1:
		d['atype']='alloc'
	elif 'POINTER' in attr1:
		d['atype']='pointer'
	elif 'ASSUMED_SHAPE' in attr2:
		d['atype']='assumed'
	elif 'CONSTANT' in attr2:
		d['bounds']=get_bounds(info,dt=dt)
		d['atype']='explicit'
	d['ndims']=get_ndims(info,dt=dt)
	return d
		
def get_ndims(info,dt=False):
	if dt:
		x=info[1].replace("(","").strip()
	else:
		x=info[4].replace("(","").strip()
	return int(x.split()[0])
	
def get_bounds(info,dt=False):
	#Horrible but easier than splitting the nested brackets
	if dt:
		val=[int(x) for x in info[1].split("'")[1:-1:2]]
	else:
		val=[int(x) for x in info[4].split("'")[1:-1:2]]
	return val	
	
	
def get_param_val(info):
	if 'PARAMETER' in info[0]:
		x=info[4].split("'")[1::2]
		if len(x)==1:
			value=parse_single_param_val(x[0])
		else:
			#Dont do whole list as last element is array size
			value=[]
			for i in range(len(x)-1):
				value.append(parse_single_param_val(x[i]))
	return value
			
def parse_single_param_val(x):
	if '@' in x:
		sp=x.split('@')
		value=str(float(sp[0])*10**float(sp[1]))
	else:
		value=str(x)
	return value


def parse_dummy(info):
	value=False
	attr=info[0]
	if 'DUMMY' in attr:
		if ' INOUT ' in attr or ' UNKNOWN-INTENT ' in attr:
			value='inout'
		elif ' OUT ' in attr:
			value='out'
		elif ' IN ' in attr:
			value='in'
	return value
	
def parse_optional(info):
	if 'OPTIONAL' in info[0]:
		return True
	else:
		return False
		
def parse_ext_func(info):
	if 'EXTERNAL DUMMY' in info[0]:
		return True
	else:
		return False
	
def parse_derived_type(info,dt_names,dt=False):
	if dt:
		x=info[0]
	else:
		x=info[2]
	if 'DERIVED' in x:
		sx=x.split()
		#Map id to parent derived type
		return map_id_dt(int(sx[1]),dt_names)
	return False	
	
def map_id_dt(i,dt_names):
	for j in dt_names:
		if j['num']==i:
			return j['name']
	raise ValueError("Cant find derived type "+str(i))
			
def split_list_dt(list_dt):
	res=[]
	for i in split_brackets(list_dt[1:-1],remove_b=True):
		x=i.split("'")
		res.append({'name':x[1],'module':x[3],'num':int(x[4].strip())})
	return res
	
def get_child_num(info):
	x=info[3][1:-1]
	return [int(y) for y in x.split()]
	
	
def processVar(obj,dt_names):
	#Get python and ctype
	obj['pytpe'],obj['ctype']=parse_type(obj['info'])	
	#Handle arrays, obj['array']==False if not an array
	obj['array']=parse_array(obj['info'])
	#Handle derived types:
	obj['dt']=parse_derived_type(obj['info'],dt_names)
	#Dont need the info list anymore
	obj.pop('info',None)
	#Or the numbers
	obj.pop('parent',None)
	obj.pop('num',None)
	return obj
	
def processParam(obj):
	obj['pytpe'],obj['ctype']=parse_type(obj['info'])	
	obj['value']=get_param_val(obj['info'])
	#Dont need the info list anymore
	obj.pop('info',None)
	#Or the numbers
	obj.pop('parent',None)
	obj.pop('num',None)
	return obj
	
def processFuncArg(obj,dt_names):
	#Get python and ctype
	obj['pytpe'],obj['ctype']=parse_type(obj['info'])	
	#Handle arrays, obj['array']==False if not an array
	obj['array']=parse_array(obj['info'])
	#Get Intents
	obj['intent']=parse_dummy(obj['info'])	
	#Is optional?
	obj['opt']=parse_optional(obj['info'])	
	#Is actualy a function being passed?
	obj['ext_func']=parse_ext_func(obj['info'])	
	#Handle derived types:
	obj['dt']=parse_derived_type(obj['info'],dt_names)	
	#Its off by one
	obj['parent']=obj['parent']-1
	#Dont need the info list anymore
	obj.pop('info',None)
	obj.pop('module',None)
	return obj
	
def processFunc(obj):
	obj['pytpe'],obj['ctype']=parse_type(obj['info'])	
	obj['arg_nums']=get_child_num(obj['info'])
	obj['args']=[]
	#Dont need the info list anymore
	obj.pop('info',None)
	obj.pop('module',None)
	obj.pop('parent',None)
	return obj
	
def processDT(obj,dt_names):
	obj['info']=split_info(obj['info'])	
	for i in obj['args']:
		i['info']=split_info(i['info'])	
		#Get python and ctype
		i['pytpe'],i['ctype']=parse_type(i['info'],dt=True)	
		#Handle arrays, i['array']==False if not an array
		i['array']=parse_array(i['info'],dt=True)
		#Handle derived types:
		i['dt']=parse_derived_type(i['info'],dt_names,dt=True)
		#Dont need the info list anymore
		i.pop('info',None)
		#Or the numbers
		i.pop('parent',None)
		i.pop('num',None)
	obj.pop('info',None)	
	return obj
	
def mapArgs2Func(funcs,func_args):
	#Find when functions have no arguments
	#Do it this way to avoid loops over func and func_args at the same time
	no_args=[]
	for idx,i in enumerate(funcs):
		if  len(i['arg_nums'])==0:
			no_args.append(idx)
		
	func_arg_par=[0]
	count=0
	for i in range(1,len(func_args)):
		if count in no_args:
			count=count+1
		p=func_args[i]['parent']
		if p is not func_args[i-1]['parent']:
			count=count+1
		#This is the index into funcs for each func_args
		func_arg_par.append(count)
		
	for idx,i in enumerate(func_args):
		funcs[func_arg_par[idx]]['args'].append(i)
		
	return funcs
	
def getDTNames(names):
	x=names.split("'")[1:]
	dt=[]
	for i,j in zip(x[0::4],x[3::4]):
		dt.append({'name':i,'num':int(j.replace("(","").replace(")",""))})
	return dt
	
	
def doStuff(filename):
	data,mod_data=load_data(filename)
	
	header=data[-1][1:-1].replace("'","").split()
	
	names=[]
	num1=[]
	num2=[]
	
	for i in range(len(header)//3):
		names.append(header[i*3])
		num1.append(header[i*3+1])
		num2.append(header[i*3+2])
	
	dt_names=getDTNames(data[2])
	
	main_data=data[6]
	split_data=re.split("([0-9]+\s\'[a-zA-Z_][\w]*?\'.*?)",main_data)
	
	dt_defs=[]
	funcs=[]
	func_args=[]
	mod_vars=[]
	module=[]
	param=[]
	
	if split_data[0]=='(':
		split_data.pop(0)
		
	for i,j in zip(split_data[0::2],split_data[1::2]):
		#Skip intrinsic functions and interfaces
		if "'(intrinsic)'" in j or 'ABSTRACT' in j:
			continue
		
		if j[1]=="(":
			#These are the components of a derived type and allways come after the
			#initial defintion of the derived type
			sp=i.split()
			dt_defs[-1]['args'].append({'num':int(sp[0]),'name':sp[1],'info':j})
		else:
			line=i+j
			sp=line.split()
			d={'num':int(sp[0]),
						'name':sp[1].replace("'",""),
						'module':sp[2].replace("'",""),
						#'unknown':sp[3].replace("'",""),
						'parent':int(sp[4]),
						'info':' '.join(sp[5:])}
						
			if 	d['parent']>1:
				d['info']=split_info(d['info'])
				func_args.append(processFuncArg(d,dt_names))	
			elif 'VARIABLE' in j:	
				d['info']=split_info(d['info'])
				mod_vars.append(processVar(d,dt_names))
			elif 'DERIVED' in j:
				#We dont call processDT as we still have to append the contents at the end
				d['args']=[]
				dt_defs.append(d)		
			elif 'PROCEDURE' in j:
				if 'UNKNOWN-PROC' in j:
					#Lower case derived type definitions
					continue
				d['info']=split_info(d['info'])
				funcs.append(processFunc(d))
			elif 'PARAMETER' in j:
				d['info']=split_info(d['info'])
				param.append(processParam(d))
			elif 'MODULE' in j:
				d['info']=split_info(d['info'])
				module.append(d)
			else:
				print(d)
				print(j)
				raise ValueError("Can't match type")
	
	#Handle DT's			
	for j in dt_defs:
		j=processDT(j,dt_names)
		
	#Map function arguments to functions
	funcs=mapArgs2Func(funcs,func_args)
	
	return mod_data,mod_vars,param,funcs,dt_defs
	
def output(filename,*args):
	with open(outname,'wb') as f:
		for i in args:
			pickle.dump(i,f)
	
	
#################################
version=1
if __name__ == "__main__":
	if len(sys.argv[1:])>0:
		files=sys.argv[1:]
	else:
		#files=[os.path.expandvars('$MESA_DIR/star/make/star_lib.mod')]
		files=['./tester.mod']
	for filename in files:
		#filename=os.path.expandvars('$MESA_DIR/star/make/star_lib.mod')
		mod_data,mod_vars,param,funcs,dt_defs=doStuff(filename)
		
		outname=mod_data['orig_file'].split('.')[0]+'.fpy'
		output(outname,version,mod_data,mod_vars,param,funcs,dt_defs)
	
			
