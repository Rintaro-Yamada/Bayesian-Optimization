#default:
#	python GPy_real_multibo.py 5 "sum_mes_noncost"

default:
	python multi_taskBO.py 30 mes
	python multi_taskBO.py 30 sum_mes
	python multi_taskBO.py 30 random