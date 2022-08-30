#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<sstream>
#include<functional>

using namespace std;
using namespace Eigen;

void d3_stage2( double* h1, double* h2, double* h3, double* v, int n_po):
    // construt 3d Hamiltonian matrix, plus potential at diag and eigen
    MatrixXd H(n_po*n_po*n_po, n_po*n_po*n_po);
    MatrixXd wf(1,1);
    VectorXd E(1);
    cout << "creat H begin\n";
    int i,j;
    for (int i1 = 0; i1<n_po; i1++ ){
        for (int i2 = 0; i2<n_po; i2++ ){
            for (int i3 = 0; i3<n_po; i3++ ){
                for (int j1 = 0; j1<n_po; j1++ ){
                    for (int j2 = 0; j2<n_po; j2++ ){
                        for (int j3 = 0; j3<n_po; j3++ ){
                            i = get_index_3d(i1,i2,i3, n_po);
                            j = get_index_3d(j1,j2,j3, n_po);
                            //cout << "begin of T" << i << " " << j << "\n" ;
                            //cout << "Hij"<< H(i,j)<< endl;
                            //cout << "H1_ij" << hCc(i1,j1)<< endl;
                            //cout << "H2_ij" << hCh(i2,j2)<< endl;
                            //cout << "H3_ij" << hCn(i3,j3)<< endl;
                            // Is there any simple methmatical form?
                            H(i,j) = h1(i1,j1)*delta(i2,j2)*delta(i3,j3) + 
                                           h2(i2,j2)*delta(i1,j1)*delta(i3,j3) +
                                           h3(i3,j3)*delta(i1,j1)*delta(i2,j2);
                            //cout << "end of T" << i << " " << j << "\n" ;
                            if (i==j){
                                H(i,j)+= v[i];
                            }
                        }
                    }
                }
            }
        }
    }

    SelfAdjointEigenSolver<MatrixXd> H_es(H);
    E=H_es.eigenvalues();
    wf=H_es.eigenvectors();
    const double e_opt = -132.58062407; // a.u.
    for(int i=0; i!=10;++i) outfile <<setw(20)<<(E(i)- e_opt)*hartreeincm<<endl;
    cout << "Transitions (per cm):" <<endl;
    for(int i=0; i!=10;++i){
        cout <<i<<"   "<<(E(i)-E(0))*hartreeincm<<endl;
    }
    cout << endl;
