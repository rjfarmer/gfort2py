import ctypes

def get_symtrees(filename):
	with open(filename,'r') as f:
		filelines=f.readlines()
	
	#Find places with no indentation
	counts=[len(l)-len(l.lstrip()) for l in filelines]
	zeros=[idx for idx,x in enumerate(counts) if x==0 ]
	
	filelines=[l.rstrip() for l in filelines]
	
	data_blocks=[]
	
	for i in range(len(zeros)-1):
		data_blocks.append(filelines[zeros[i]:zeros[i+1]])
	
	data_blocks.append(filelines[zeros[-1]:])
	
	#Split each data_block into its component symtrees
	symtrees=[]
	for i in data_blocks:
		x=parse_data_block(i)
		if len(x[0])>0:
			symtrees.append(x)
			
	return symtrees


def split_brackets(value):
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
			res.append(token[1:-1])
			start=False
			token=''
	return res
	
def parse_data_block(data):
	'''
	Split a top level block, i.e something that was at 0 identation,
	into its symtree ( list of dicts containinf info on varaibles and functions
	
	'''
	dout=[]
	for l in data:
		if 'code:' in l:
			break
		if len(l.strip())==0 or 'CONTAINS' in l or 'Namespace' in l:
			continue
		dout.append(l)
	if len(dout)==0:
		return '',[]

	proc_name=dout[0].split('=')[-1].strip()
	symtree=[]
	for idx,l in enumerate(dout):
		if 'symtree:' in l:
			name=l.split('symtree:')[1].split('||')[0].strip().strip("'")
			if '__' in name:
				continue
			symtree.append({})
			symtree[-1]['name']=name
			symtree[-1]['line']=l
			i=idx
			while True:
				i=i+1
				l2=dout[i]
				if 'symtree:' in l2:
					break
				name,val=l2.split(':')
				symtree[-1][name.strip().strip("'")]=val.strip().strip("'")
				if i == len(dout)-1:
					break
	return proc_name,symtree
	
def parse_type_spec(spec):
	'''
	Determine the variable type and byte size 
	
	InputL
	      type spec : (INTEGER 4)

	Returns:
		{'type':'int',size:'4'}
	'''
	res={'type':'void','size':0}
	
	if 'INTEGER' in spec:
		res['type']='int'
	elif 'REAL' in spec:
		res['type']='float'
	elif 'DERIVED' in spec:
		res['type']='struct'
	elif 'COMPLEX' in spec:
		res['type']='complex'
	elif 'UNKNOWN' in spec:
		res['type']='void'
	elif 'CHARACTER' in spec:
		res['type']='char'
	elif 'LOGICAL' in spec:
		res['type']='bool'
	else:
		raise ValueError("Cant parse "+spec)

	res['size']=spec.split()[-1][0:-1]

	return res
	
	
def parse_value(value):
	"""
	Determine values for parameters, arrays start are enclosed in (/ /)
	"""
	if '(/' in value:
		#Array
		res=value.strip('(/').strip('/)').split(',')
		res=[i.split('_')[0] for i in res]
	else:
		res=value
	return res

def parse_array_spec(value):
	#(1 [0] AS_EXPLICIT 1 5 ) or (1 [0] AS_DEFERRED () () ) or (1 [0] AS_ASSUMED_SHAPE 1 () ) 
	#ndims something AS_EXPLICIT|AS_DEFERERED|AS_ASSUMED_SHAPE l|ubound_1 l|ubound_2 ....
	res={}
	if len(value)==0 or value=='()':
		return res
	
	res['ndims']=value.split()[0][1:]
	if "AS_EXPLICIT" in value:
		res['allocatable']=False
		res['shape']=value.split()[3:-1]
	elif 'AS_DEFERRED' in value:
		res['allocatable']=False
		res['shape']=-1
	elif 'AS_ASSUMED_SHAPE' in value:
		res['allocatable']=True
		res['shape']=-1
	else:
		raise ValueError("Bad array_spec "+value)
	
	return res
	
def parse_struct_comp(value):	
	res=split_brackets(value)
	res2=[]
	for i in res:
		val={}
		val['name']=i.split(' ')[0]
		typ,arr=split_brackets(i)
		val['type_spec']=parse_type_spec('('+typ+')')
		val['array_spec']=parse_array_spec('('+arr+')')
		res2.append(val)
	return res2

def parse_dummy(value):
	#In/Out
	if 'DUMMY(IN)' in value:
		inout='in'
	elif 'DUMMY(INOUT)' in value:
		inout='inout'
	elif 'DUMMY(OUT)' in value:
		inout='out'
	elif 'DUMMY)' in value:
		inout='inout'
	else:
		inout=False
	return inout

def get_functions(symtree_head):
	# List of fucntions from the top level of the module
	res=[]
	for val in symtree_head:
		if 'attributes' in val.keys():
			if 'PROCEDURE MODULE-PROC' in val['attributes']:
				res.append(val['name'])
	return res
	
def get_func_return(func_name,symtree_head):
	for val in symtree_head:
		if func_name==val['name']:
			res=parse_type_spec(val['type spec'])
	return res

def get_vars(symtree):
	res=[]
	
	for val in symtree:
		if 'attributes' in val.keys():
			if 'VARIABLE' in val['attributes'] or 'PARAMETER' in val['attributes'] or 'DUMMY-PROC' in val['attributes']:
				var={}
				var['name']=val['name']
				var['type_spec']=parse_type_spec(val['type spec'])
				
				var['value']={}
				if 'value' in val.keys() and 'DERIVED' not in val['type spec']:
					var['value']=parse_value(val['value'])
					
				var['param']=False
				if 'PARAMETER' in val['attributes']:
					var['param']=True
					
				var['array']=False
				var['array_spec']={}
				if 'DIMENSION' in val['attributes']:
					var['array']=True
					var['array_spec']=parse_array_spec(val['Array spec'])
					
				var['struct']=False
				#Instance of a structure
				var['struct_decl']=None
				if 'DERIVED' in val['type spec']:
					var['struct']=True
					var['struct_decl']=var['type_spec']['size']
					
				var['pointer']=False
				if 'POINTER' in val['attributes']:
					var['pointer']=True
					
				var['dummy']=parse_dummy(val['attributes'])
				
				var['is_func']=False
				if 'DUMMY-PROC' in val['attributes']:
					var['is_func']=True
							
				res.append(var)
	return res
	
def get_struct_defs(symtree_head):
	res=[]
	for val in symtree_head:
		if 'DERIVED' in val['attributes'] and 'UNKNOWN' in val['type spec']:
			var={}
			var['name']=val['name']
			var['struct_def']=parse_struct_comp(val['components'])
			res.append(var)
	return res

def get_funcs(symtrees_all):
	res=[]
	
	symtrees=symtrees_all[1:]
	symtree_head=symtrees_all[0][1]
	func_names=get_functions(symtree_head)
	
	for i in func_names:
		for j in symtrees[1:]:
			if i == j[0]:
				val={}
				val['name']=i
				args=get_vars(j[1])
				#remove non dummy arguments (aka local varaibles) which are False
				args=[x for x in args if x['dummy']]
				#Sort into the actual order
				arg_order=get_func_arg_order(i,symtree_head)
				order_dict = {arg: index for index, arg in enumerate(arg_order)}
				args.sort(key=lambda x: order_dict[x["name"]])	
				val['args']=args
				
				#Get function return type
				val['return']=get_func_return(i,symtree_head)
				
				res.append(val)
	return res
	
def get_func_arg_order(func_name,symtree_head):
	res=[]
	for j in symtree_head:
		if func_name == j['name']:
			return j['Formal arglist'].split()


#######################################################################
# ctype handling
#######################################################################

def mangle_name(mod,name):
	return "__"+mod+'_MOD_'+name

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
		raise ValueError("Cant find suitable int for size "+size)	
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
		raise ValueError("Cant find suitable float for size"+size)

	return res
	
def get_ctype_bool(size):
	return 'c_bool'	

def get_ctype_str(size):
	return 'c_char_p'	

def map_to_scalar_ctype(var):
	"""
	gets the approitate ctype for a variable
	
	Returns:
		String
	"""
	typ=var['type_spec']['type']
	size=var['type_spec']['size']

	res=None
	if typ=='int':
		res=get_ctype_int(size)
	elif typ=='float':
		res=get_ctype_float(size)	
	elif typ=='char':
		res=get_ctype_str(size)
	elif typ=='bool':
		res=get_ctype_bool(size)
	elif typ=='struct':
		raise ValueError("Should of called map_to_struct_ctype")
	else:
		raise ValueError("Not supported ctype "+var['name']+' '+str(typ)+' '+str(size))
	
	return res
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
#######################################################

if __name__ == '__main__':
	#Compile code with -fdump-fortran-original and pipe output to file.fpy
	filename='./test_mod.fpy'
	
	symtrees=get_symtrees(filename)
	
	module_name=symtrees[0][0]
	mod_vars=get_vars(symtrees[0][1])
	struct_defs=get_struct_defs(symtrees[0][1])
	mod_funcs=get_funcs(symtrees)			



	

