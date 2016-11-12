import gzip


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
		#if module varaible then is the module name else its an empty string
		res[-1]['module_name']=dsplit[i+2].replace("'",'')
		res[-1]['term2']=dsplit[i+3].replace("'",'')
		res[-1]['parent_num']=dsplit[i+4].replace("'",'')
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
	for i in list_dicts:
		if i[key]==value:
			print(i)
			
def clean_list(l,idx):
	return [i for j, i in enumerate(l) if j not in idx]
			
def get_all_head_objects(data):
	object_head=parse_object_names(data)
	object_all=parse_all_objects(data)
	#Maps function attributes to the names
	for idx,i in enumerate(object_all):
		for j in object_head:
			if i['num']==j['num']:
				#merge dicts
				j.update(i)
				j['arg_nums']=j['attr'][3].split()
	
	get_func_args(object_head,object_all)
				
	return object_head,object_all
	
def get_func_args(object_head,object_all):
	ind=[]
	for idx,i in enumerate(object_all):
		for j in object_head:
			if i['num'] in j['arg_nums']:
				j['args'].append(i)
				ind.append(idx)
	
def parse_type(obj):
	attr=obj['attr'][2]
	res=''
	if 'INTEGER' in attr:
		res='int'
	elif 'REAL' in attr:
		res='float'
	elif 'DERIVED' in attr:
		res='struct'
	elif 'COMPLEX' in attr:
		res='complex'
	elif 'UNKNOWN' in attr:
		res='void'
	elif 'CHARACTER' in attr:
		res='char'
	elif 'LOGICAL' in attr:
		res='bool'
	else:
		raise ValueError("Cant parse "+attr)
	obj['ctype']=res
	
def parse_type_size(obj):
	obj['csize']=obj['attr'][2].split()[1]
	
def get_type_spec(obj):
	parse_type(obj)
	parse_type_size(obj)
	
def get_obj_type(obj):
	x=obj['attr'][0]
	if 'DERIVED' in x:
		x['is_derived']=True
	if 'VARIABLE' in x:
		x['is_var']=True
	if 'SUBROUTINE' in x:
		x['is_sub']=True
	if 'FUNCTION' in x:
		x['is_func']=True
	if 'MODULE ' in x:
		x['is_module']=True	

if __name__=='__main__':
	#filename='/media/data/mesa/mesa/dev/star/make/star_lib.mod'
	filename='tester.mod'
	data,mod_data=load_data(filename)
	obj_head_all,object_all=get_all_head_objects(data)

	for i in obj_head_all:
		get_type_spec(i)

