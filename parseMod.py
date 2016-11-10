import ctypes
import pickle
import sys
import gzip

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
			res.append(token)
			start=False
			token=''
	return res

#Mix of function names, module variables, parameters and possible 
#fortran intrinsic functions
def object_names(data):
	y=data[-1].split()
	mod=[]
	for i in range(0,len(y),3):
		name=y[i].replace('(','').replace("'",'')
		if not name.startswith('__'):
			mod.append({})
			mod[-1]['name']=name
			mod[-1]['ambigous']=y[i+1]
			mod[-1]['num']=int(y[i+2].replace(')',''))
	return mod

def get_all_objects(data):
	#Remove opening and closing bracket
	dsplit=data[-2][1:-1].split(' ')
	#Pattern is 4 terms then look for matching set of open/close brackets
	res=[]
	i=0
	while True:
		if i >= len(dsplit)-1:
			break
		res.append({})
		res[-1]['num']=int(dsplit[i].replace("'",''))
		res[-1]['name']=dsplit[i+1].replace("'",'')
		#if module varaible then is the module name else its an empty string
		res[-1]['module_name']=dsplit[i+2].replace("'",'')
		res[-1]['term2']=dsplit[i+3].replace("'",'')
		res[-1]['parent_num']=int(dsplit[i+4].replace("'",''))
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
		res[-1]['attr']=token
		i=i+5+count2
	return res
	
def load_data(filename):
	with gzip.open(filename) as f:
		x=f.read()
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
	

if __name__=='__main__':
	filename='tester.mod'
	data,mod_data=load_data(filename)
	object_head=object_names(data)
	object_all=get_all_objects(data)
	
