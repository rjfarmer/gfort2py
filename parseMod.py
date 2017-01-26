import gzip
import ctypes
import os
import pickle
import sys

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

#Mix of function names, module variables, parameters and possible 
#fortran intrinsic functions
def parse_object_names(data):
	y=data[-1].split()
	mod=[]
	for i in range(0,len(y),3):
		name=y[i].replace('(','').replace("'",'')
		if not name.startswith('__'):
			mod.append({})
			mod[-1]['name']=name
			mod[-1]['ambiguous']=y[i+1]
			mod[-1]['num']=y[i+2].replace(')','')
			mod[-1]['args']=[]
	return mod

def parse_all_objects(data):
	#Remove opening and closing bracket
	dsplit=data[-2][1:-1].split(' ')
	#Pattern is 4 terms then look for matching set of open/close brackets
	res=[]
	i=0
	while True:
		if i >= len(dsplit)-1:
			break
		res.append({})
		res[-1]['num']=dsplit[i].replace("'",'')
		res[-1]['name']=dsplit[i+1].replace("'",'')
		#if module variable then is the module name else its an empty string
		res[-1]['module_name']=dsplit[i+2].replace("'",'')
		res[-1]['term2']=dsplit[i+3].replace("'",'')
		res[-1]['parent_num']=dsplit[i+4].replace("'",'')
		res[-1]['mangled_name']=mangle_name(res[-1])
		#Look for an open and closed bracket pair
		token=''
		start=False
		count=0
		count2=0
		for j in dsplit[i+5:]:
			count2=count2+1
			count=count+j.count('(')-j.count(')')
			if j.count('('):
				start=True
			if start:
				token+=str(j)+' '
			if start and count==0:
				start=False
				break
		res[-1]['attr']=split_brackets(token[1:-1],remove_b=True)
		i=i+5+count2
	return res
	
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
	
def find_key_val(list_dicts,key,value):
	for idx,i in enumerate(list_dicts):
		if i[key]==value:
			print(idx)
			print(i)
			
def clean_list(l,idx):
	return [i for j, i in enumerate(l) if j not in idx]
	
			
def get_all_head_objects(data):
	object_head=parse_object_names(data)
	object_all=parse_all_objects(data)
	#Maps function attributes to the names
	for j in object_head:
		for idx,i in enumerate(object_all):
			#functions
			if i['num']==j['num']:
				#merge dicts
				j.update(i)
				j['arg_nums']=j['attr'][3].split()
		parse_type(j)
		parse_array(j)
		parse_struct_types(j,object_head)
		parse_derived_type_def(j,object_head)
	
	get_func_args(object_head,object_all)
			
	return object_head
	
def get_func_args(object_head,object_all):
	ind=[]
	for j in object_head:
		for idx,i in enumerate(object_all):
			if i['num'] in j['arg_nums']:
				j['args'].append(process_func_arg(i,object_head))
				ind.append(idx)
	
def process_func_arg(obj,object_head):
	parse_type(obj)
	parse_array(obj)
	parse_dummy(obj)
	parse_struct_types(obj,object_head)
	clean_dict_func_arg(obj)
	return obj


def parse_derived_type_def(obj,object_head):
	#This should be the parent defintion of a derived type
	if obj['parent_num'] != 1 and 'DERIVED' not in obj['attr'][0]:
		return
	
	elem=split_brackets(obj['attr'][1])
	obj['args']=[]
	for i in elem:
		j={}
		x=split_brackets(i[1:-1])
		j['name']=i.split()[1]
		j['num']=i.split()[0].replace('(','')
		j['attr']=[x[2][1:-1],0,x[0][1:-1],0,x[1][1:-1]]
		obj['args'].append(process_func_arg(j,object_head))
		
	clean_dict(obj,['attr','term2'])

def parse_struct_types(obj,object_head):
	if 'DERIVED' in obj['attr'][0]:
	#This is the definition of the derived type not a variable of type derived type
		return
	for j in object_head:
		if 'DERIVED' in obj['attr'][2] and obj['attr'][2].split()[1]==j['num']:
			obj['struct_type']=j['name']
			break	

def clean_dict_func_arg(obj):
	# remove uneeded entries from a function arg 
	# (may need some of these if its not a func arg)
	remove=['attr','term2','parent_num','module_name']
	clean_dict(obj,remove)

def clean_mod(obj):
	remove=['attr','term2','parent_num','ambiguous']
	clean_dict(obj,remove)
	
def clean_dict(obj,names):
	for i in names:
		obj.pop(i,None)		
	
def parse_type(obj):
	get_param_val(obj)
	attr=obj['attr'][2]
	obj['csize']=obj['attr'][2].split()[1]
	if 'INTEGER' in attr:
		obj['type']='int'
		obj['ctype']=get_ctype_int(obj['csize'])
	elif 'REAL' in attr:
		obj['type']='float'
		obj['ctype']=get_ctype_float(obj['csize'])
	elif 'DERIVED' in attr:
		obj['ctype']='struct'
		obj['type']='void'
	elif 'COMPLEX' in attr:
		obj['type']='void'
		obj['ctype']=get_ctype_float(obj['csize'])
	elif 'CHARACTER' in attr:
		obj['ctype']='char'
		obj['type']='str'
	elif 'LOGICAL' in attr:
		obj['ctype']='bool'
		obj['type']='bool'
	elif 'UNKNOWN' in attr:
		obj['ctype']='void'
		obj['type']='void'
	else:
		raise ValueError("Cant parse "+attr)
	
def get_ctype_int(size):
	res=None
	size=int(size)
	if size==ctypes.sizeof(ctypes.c_int):
		res='c_int'
	elif size==ctypes.sizeof(ctypes.c_int16):
		res='c_int16'
	elif size==ctypes.sizeof(ctypes.c_int32):
		res='c_int32'
	elif size==ctypes.sizeof(ctypes.c_int64):
		res='c_int64'
	else:
		raise ValueError("Cant find suitable int for size "+str(size))	
	return res
	
def get_ctype_float(size):
	res=None
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
	
def parse_array(obj):
	obj['array']=False
	obj['ndims']=-1
	obj['bounds']=[]
	obj['atype']=None
	if 'DIMENSION' not in obj['attr'][0]:
		return
	if 'ALLOCATABLE' in obj['attr'][0]:
		obj['array']=True
		obj['ndims']=get_ndims(obj)
		obj['atype']='alloc'
	elif 'POINTER' in obj['attr'][0]:
		obj['array']=True
		obj['ndims']=get_ndims(obj)
		obj['atype']='pointer'
	elif 'ASSUMED_SHAPE' in obj['attr'][4]:
		obj['array']=True
		obj['ndims']=get_ndims(obj)
		obj['atype']='assumed'
	elif 'CONSTANT' in obj['attr'][4]:
		obj['array']=True
		obj['ndims']=get_ndims(obj)
		obj['bounds']=get_bounds(obj)
		obj['atype']='explicit'
		
def get_ndims(obj):
	return int(obj['attr'][4].split()[0])
	
def get_bounds(obj):
	#Horrible but easier than splitting the nested brackets
	return [int(x) for x in obj['attr'][4].split("'")[1:-1:2]]
	
def parse_dummy(obj):
	attr=obj['attr'][0]
	
	if 'DUMMY' in attr:
		if 'INOUT' in attr or 'UNKNOWN-INTENT' in attr:
			obj['inout']='inout'
		elif 'OUT ' in attr:
			obj['inout']='out'
		elif 'IN ' in attr:
			obj['inout']='in'
		else:
			obj['inout']=False

def mangle_name(obj):
	return '__'+obj['module_name']+'_MOD_'+obj['name']
	
def get_param_val(obj):
	if 'PARAMETER' in obj['attr'][0]:
		obj['param']=True
		x=obj['attr'][4].split()[-1].replace("'",'')
		if '@' in x:	
			obj['value']=str(float(x.split('@')[0])*10**float(x.split('@')[1]))
		else:
			obj['value']=str(x)




if len(sys.argv[1:])>0:
	files=sys.argv[1:]
else:
	#files=[os.path.expandvars('$MESA_DIR/star/make/star_lib.mod')]
	files=['./tester.mod']


for filename in files
	data,mod_data=load_data(filename)
	obj_head_all=get_all_head_objects(data)
	
	#cleanup unneed entries
	for i in obj_head_all:
		clean_dict(i,['num','parent_num','ambiguous','arg_nums','term2','attr'])
	
	#Save data
	outname=mod_data['orig_file'].split('.')[0]+'.fpy'
	
	version=1
	
	with open(outname,'wb') as f:
		pickle.dump(version,f)
		pickle.dump(mod_data,f)
		pickle.dump(obj_head_all,f)
	
	data=0
	mod_data=0
	obj_head_all=0
