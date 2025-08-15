import subprocess

policy = ["numactl", "--interleave=all"]

matrices=[
	[["audikw_1.mtx", "1.1"],   ["7","4","2"]],
	[["Bump_2911.mtx", "1.1"],  ["8","5","2"]],
	[["ecology2.mtx", "1.0"],   ["8","5","2"]],
	[["Emilia_923.mtx", "1.0"], ["8","5","2"]],
	[["G3_circuit.mtx", "1.0"], ["9","4","2"]],
	[["hpcg_7_7_7.mtx", "1.0"], ["9","4","2"]],
	[["hpcg_8_7_7.mtx", "1.0"], ["6","4","2"]],
	[["hpcg_8_8_7.mtx", "1.0"], ["7","4","2"]],
	[["hpcg_8_8_8.mtx", "1.0"], ["6","4","2"]],
	[["ldoor.mtx", "1.1"],      ["8","5","2"]],
	[["Queen_4147.mtx", "1.1"], ["8","4","2"]],
	[["Serena.mtx", "1.1"],     ["8","6","2"]],
	[["thermal2.mtx", "1.0"],   ["9","4","2"]],
	[["tmt_sym.mtx", "1.0"],    ["8","6","2"]],
	[["atmosmodd.mtx", "1.0"],   	 ["8","6","2"]],
	[["atmosmodj.mtx", "1.0"],   	 ["9","4","2"]],
	[["atmosmodl.mtx", "1.0"],   	 ["8","3","2"]],
	[["hpgmp_7_7_7.mtx", "1.0"], 	 ["8","4","1"]],
	[["hpgmp_8_7_7.mtx", "1.0"], 	 ["8","4","1"]],
	[["hpgmp_8_8_7.mtx", "1.0"], 	 ["8","4","1"]],
	[["hpgmp_8_8_8.mtx", "1.0"], 	 ["6","4","2"]],
	[["ss.mtx", "1.1"],            ["8","4","2"]],
	[["stokes.mtx", "1.0"],        ["8","4","1"]],
	[["t2em.mtx", "1.0"],          ["6","4","2"]],
	[["tmt_unsym.mtx", "1.0"],     ["10","4","2"]],
	[["Transport.mtx", "1.0"],   	 ["8","3","2"]],
	[["vas_stokes_1M.mtx", "1.0"], ["8","4","1"]],
	[["vas_stokes_2M.mtx", "1.0"], ["8","4","1"]],
]

def T3CP(ave, data, switch):
	results = []
	if switch == 3: # For figure 3
		exe = "./bin/f3r16.exe"
		params = [
			["8","4","1"], ["8","4","2"], ["8","4","3"], ["8","4","4"],
			["8","2","2"], ["8","3","2"], ["8","5","2"], ["8","6","2"],
			["6","4","2"], ["7","4","2"], ["9","4","2"], ["10","4","2"]
		]
		for p in params:
			add = p + ["64", "3"]
			result = subprocess.run(policy+[exe]+data[0]+[str(ave)]+add, capture_output=True, text=True)
			results.append(result.stdout)
	
	if switch == 4: # For figure 4
		exes = [
			"./bin/f3r16.exe",
			"./bin/f2.exe", './bin/f2h.exe',
			"./bin/f3.exe", './bin/f3h.exe',
			"./bin/f4.exe"
		]
		for b in exes:
			add = None
			if 'f3r' in b:
				add = ["8", "4", "2", "64", "3"]
			else:
				add = ["0", "0", "0", "0", "3"]
			result = subprocess.run(policy+[b]+data[0]+[str(ave)]+add, capture_output=True, text=True)
			results.append(result.stdout)

	if switch == 5: # For figure 5
		exes = [
			"./bin/f3r16_log.exe",
			"./bin/f2_log.exe", './bin/f2h_log.exe',
			"./bin/f3_log.exe", './bin/f3h_log.exe',
			"./bin/f4_log.exe"
		]
		for b in exes:
			add = None
			if 'f3r' in b:
				add = ["8", "4", "2", "64", "3"]
			else:
				add = ["0", "0", "0", "0", "3"]
			result = subprocess.run(policy+[b]+data[0]+[str(ave)]+add, capture_output=True, text=True)
			results.append(result.stdout)

	if switch == 6: # For figure 6
		exe = "./bin/f3r16.exe"
		params = ["1", "4", "16", "32", "64", "128", "256"]
		for p in params:
			add = ["8","4","2"] + [p] + ["3"]
			result = subprocess.run(policy+[exe]+data[0]+[str(ave)]+add, capture_output=True, text=True)
			results.append(result.stdout)

	if switch == 7: # For figure 7
		exe = "./bin/static.exe"
		params = ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"]
		result = subprocess.run(policy+["./bin/f3r16.exe"]+data[0]+[str(ave)]+["8","4","2","64","3"], capture_output=True, text=True)
		results.append(result.stdout)
		for p in params:
			add = ["8","4","2","0","3"] + [p]
			result = subprocess.run(policy+[exe]+data[0]+[str(ave)]+add, capture_output=True, text=True)
			results.append(result.stdout)
	
	return results

import sys

if __name__ == '__main__':
	average = int(sys.argv[1])

	if len(sys.argv) > 3:
		if sys.argv[3] == "full":
			step=1
		else:
			print("argv[3] must be full")
			exit(1)
	else:
		step=3

	if sys.argv[2] == "figure3":
		with open("t3c-figure3.csv", mode="w") as f:
			f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
			for data in matrices[::step]:
				results = T3CP(average, data, 3)
				for res in results:
					if res != '\n':
						f.write(res)
				f.flush()

	if sys.argv[2] == "figure4":
		with open("t3c-figure4.csv", mode="w") as f:
			f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
			for data in matrices[::step]:
				results = T3CP(average, data, 4)
				for res in results:
					if res != '\n':
						f.write(res)
				f.flush()

	if sys.argv[2] == "figure5":
		subset = [
			[["atmosmodd.mtx", "1.0"],   	 ["8","6","2"]],
			[["vas_stokes_2M.mtx", "1.0"], ["8","4","1"]],
		]
		with open("t3c-figure5.csv", mode="w") as f:
			for data in subset:
				results = T3CP(average, data, 5)
				for res in results:
					if res != '\n':
						f.write(res)
				f.flush()

	if sys.argv[2] == "figure6":
		with open("t3c-figure6.csv", mode="w") as f:
			f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
			for data in matrices[::step]:
				results = T3CP(average, data, 6)
				for res in results:
					if res != '\n':
						f.write(res)
				f.flush()

	if sys.argv[2] == "figure7":
		with open("t3c-figure7.csv", mode="w") as f:
			f.write("Problem,Method,Prec,M2,M3,M4,W,Precond,ACC,Time,Iter,ImplRes,ExplRes\n")
			for data in matrices[::step]:
				results = T3CP(average, data, 7)
				for res in results:
					if res != '\n':
						f.write(res)
				f.flush()
				