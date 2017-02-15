import numpy as np
import gfort2py as gf


x=gf.fFort('./tester.so','tester.mod')

num=gf.find_key_val(x._param,'name','const_str')
x._init_param()
print(x._get_param(x._param[num]))
try:
	x._set_param('abcdefgdet',x._param[num])
except AttributeError:
	print("Success")
else:
	print(x._get_param(x._param[num]))


num=gf.find_key_val(x._mod_vars,'name','a_str')
x._init_var(x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))
x._set_var('abcdefgdet',x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))


num=gf.find_key_val(x._mod_vars,'name','a_int')
x._init_var(x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))
x._set_var(2,x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))

num=gf.find_key_val(x._mod_vars,'name','b_int_exp_1d')
x._init_var(x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))
x._set_array(np.array([5,6,7,8,9]),x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))

num=gf.find_key_val(x._mod_vars,'name','b_int_exp_2d')
x._init_var(x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))
x._set_array(np.zeros([5,5],dtype='int'),x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))


num=gf.find_key_val(x._funcs,'name','sub_no_args')
x._init_func(x._funcs[num])
x._call(x._funcs[num])

num=gf.find_key_val(x._funcs,'name','func_int_no_args')
x._init_func(x._funcs[num])
x._call(x._funcs[num])

num=gf.find_key_val(x._funcs,'name','func_real_no_args')
x._init_func(x._funcs[num])
x._call(x._funcs[num])

num=gf.find_key_val(x._funcs,'name','func_real_dp_no_args')
x._init_func(x._funcs[num])
x._call(x._funcs[num])



