import subprocess

symmetric=[
	# [matrix, alpha], [params for F3R-best]
	[["apache2.mtx", "1.0"],    ["8","4","2"]],
	[["audikw_1.mtx", "1.6"],   ["8","6","2"]],
	[["Bump_2911.mtx", "1.2"],  ["8","6","2"]],
	[["ecology2.mtx", "1.0"],   ["8","5","2"]],
	[["Emilia_923.mtx", "1.2"], ["8","5","2"]],
	[["G3_circuit.mtx", "1.0"], ["9","4","2"]],
	[["hpcg_7_7_7.mtx", "1.0"], ["8","6","2"]],
	[["hpcg_8_7_7.mtx", "1.0"], ["8","3","2"]],
	[["hpcg_8_8_7.mtx", "1.0"], ["6","4","2"]],
	[["hpcg_8_8_8.mtx", "1.0"], ["6","4","2"]],
	[["ldoor.mtx", "1.3"],      ["8","6","2"]],
	[["Queen_4147.mtx", "1.3"], ["10","4","2"]],
	[["Serena.mtx", "1.2"],     ["7","4","2"]],
	[["thermal2.mtx", "1.0"],   ["10","4","2"]],
	[["tmt_sym.mtx", "1.0"],    ["8","6","2"]],
]

general=[
	[["atmosmodd.mtx", "1.0"],   	 ["8","2","2"]],
	[["atmosmodj.mtx", "1.0"],   	 ["8","2","2"]],
	[["atmosmodl.mtx", "1.0"],   	 ["8","6","2"]],
	[["Freescale1.mtx", "1.1"],    ["9","4","2"]],
	[["hpgmp_7_7_7.mtx", "1.0"], 	 ["8","2","2"]],
	[["hpgmp_8_7_7.mtx", "1.0"], 	 ["6","4","2"]],
	[["hpgmp_8_8_7.mtx", "1.0"], 	 ["8","2","2"]],
	[["hpgmp_8_8_8.mtx", "1.0"], 	 ["6","4","2"]],
	[["ss.mtx", "1.2"],            ["8","6","2"]],
	[["stokes.mtx", "1.3"],        ["8","4","1"]],
	[["t2em.mtx", "1.0"],          ["9","4","2"]],
	[["tmt_unsym.mtx", "1.0"],     ["8","5","2"]],
	[["Transport.mtx", "1.0"],   	 ["9","4","2"]],
	[["vas_stokes_1M.mtx", "1.3"], ["8","4","1"]],
	[["vas_stokes_2M.mtx", "1.3"], ["8","4","1"]],
	[["rajat31.mtx", "1.0"],       ["8","4","1"]],
]

def T3G_SYM(ave, data):
	sym_bins = [
		"./bin/cg64-gpu.exe", "./bin/cg32-gpu.exe", "./bin/cg16-gpu.exe",
		"./bin/gm64-gpu.exe", "./bin/gm32-gpu.exe", "./bin/gm16-gpu.exe",
		"./bin/f3r64-gpu.exe", "./bin/f3r32-gpu.exe", "./bin/f3r16-gpu.exe",
	]
	sym_results = []
	for b in sym_bins:
		add = ["8", "4", "2", "64", "3"] if 'f3r' in b else []
		result = subprocess.run([b]+data[0]+[str(ave)]+add, capture_output=True, text=True)
		sym_results.append(result.stdout)
	add = data[1] + ["64", "3"]
	result = subprocess.run([sym_bins[-1]]+data[0]+[str(ave)]+add, capture_output=True, text=True)
	sym_results.append(result.stdout.replace('__half', 'Best'))
	return sym_results

def T3G_GEN(ave, data):
	gen_bins = [
		"./bin/bicg64-gpu.exe", "./bin/bicg32-gpu.exe", "./bin/bicg16-gpu.exe",
		"./bin/gm64-gpu.exe", "./bin/gm32-gpu.exe", "./bin/gm16-gpu.exe",
		"./bin/f3r64-gpu.exe", "./bin/f3r32-gpu.exe", "./bin/f3r16-gpu.exe",
	]
	gen_results = []
	for b in gen_bins:
		add = ["8", "4", "2", "64", "3"] if 'f3r' in b else []
		result = subprocess.run([b]+data[0]+[str(ave)]+add, capture_output=True, text=True)
		gen_results.append(result.stdout)
	add = data[1] + ["64", "3"]
	result = subprocess.run([gen_bins[-1]]+data[0]+[str(ave)]+add, capture_output=True, text=True)
	gen_results.append(result.stdout.replace('__half', 'Best'))
	return gen_results

import sys

if __name__ == '__main__':
	average = int(sys.argv[1])

	if len(sys.argv) > 2:
		if sys.argv[2] == "quick":
			step=2
		else:
			print("argv[2] should be quick")
			exit(1)
	else:
		step=1

	with open("t3g-symmetric.csv", mode="w") as f:
		f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
		for data in symmetric[::step]:
			sym_results = T3G_SYM(average, data)
			for res in sym_results:
				if res != '\n':
					f.write(res)
			f.flush()

	with open("t3g-general.csv", mode="w") as f:
		f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
		for data in general[::step]:
			gen_results = T3G_GEN(average, data)
			for res in gen_results:
				if res != '\n':
					f.write(res)
			f.flush()
