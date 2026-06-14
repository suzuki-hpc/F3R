#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

struct COO {
	std::vector<long> row;
	std::vector<long> col;
	std::vector<double> val;
	long n, nnz;
};

COO generate_hpgmp(int nx, int ny, int nz, double beta) {
	long n = nx * ny * nz;

	auto res = COO();
	res.row.resize(n * 27);
	res.col.resize(n * 27);
	res.val.resize(n * 27);
	res.n = n;

	long cnt=0;
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				long row=k*ny*nx + j*nx + i;
	      
				for (int zz=-1; zz<=1; zz++) {
					for (int yy=-1; yy<=1; yy++) {
						for (int xx=-1; xx<=1; xx++) {

							if (k+zz < 0 || j+yy < 0 || i+xx < 0)
								continue;
							if (nz <= k+zz || ny <= j+yy || nx <= i+xx)
								continue;

							long col = (k+zz)*ny*nx + (j+yy)*nx + i+xx;

							if (0 <= col && col < n) {
								res.row[cnt]=row+1;
								res.col[cnt]=col+1;
								if (row == col)
									res.val[cnt]=26;
								else {
									if (yy == 0 && xx == 0 && zz != 0) {
										if (zz == -1)
											res.val[cnt]=-1-beta;
										else if (zz == 1)
											res.val[cnt]=-1+beta;
									} else
										res.val[cnt]=-1;
								}
								cnt += 1;
							}

						}
					}
				}
			}
		}
	}
	res.nnz = cnt;
	return res;
}

void save_matrix_market(std::string filename, COO coo) {
	std::ofstream outfile(filename);

	outfile << "%%MatrixMarket matrix coordinate real general\n";
	outfile << coo.n << " " << coo.n << " " << coo.nnz << "\n";
	for (long i=0; i<coo.nnz; i++)
		outfile << coo.row[i] << " " << coo.col[i] << " " << coo.val[i] << "\n";

	outfile.close();
}

int main(int argc, char const *argv[]) {
	int x = atoi(argv[1]);
	int y = atoi(argv[2]);
	int z = atoi(argv[3]);
	int nx=std::pow(2, x);
	int ny=std::pow(2, y);
	int nz=std::pow(2, z);
	long n=nx*ny*nz;

	auto suffix= std::string(argv[1]) + "_" + argv[2] + "_" + argv[3];
	auto filename = std::string("hpgmp_") + suffix + ".mtx";
	COO coo = generate_hpgmp(nx, ny, nz, 0.5);
	save_matrix_market(filename, coo);

	return 0;
}
