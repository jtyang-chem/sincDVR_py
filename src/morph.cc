/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// By Yu Zhai <me@zhaiyusci.net>
// 3d by jtyang 2021/3/9

#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<sstream>
#include<functional>
#include"podvr.h"
#include <sys/time.h>

using namespace std;
using namespace Eigen;

inline int get_index_3d(int i, int j, int k, int nDim = 7){
// get index from 3 dimension, following the rule like quinary 
// i, cc, j, ch sym. , k, cn
// inline may let it faster
    return i*nDim*nDim+j*nDim+k;
}

inline double square(double x){ return x*x ;}
inline int delta(int i, int j){ return (i==j)?1:0;}

void wrt_eigen_m( MatrixXd m ){
    std::ofstream file("m.csv");
    IOFormat fmt(FullPrecision, 0, ",", "\n",  "","" );
    file << m.format(fmt) << endl;
}

VectorXd get_v_cross(string fin, int nP = 343){
    // get v on cross point from file

    VectorXd v_cross(nP);
    VectorXd x_cross(nP);
    ifstream f(fin, ios::in);
    for ( int i=0;i<nP;i++){
        //f >> x_cross[i] >> v_cross[i];
        f >> v_cross[i];
    }
    f.close();

    return v_cross;
}


VectorXd v_kronecker_product(const VectorXd a ,const  VectorXd b){
    int lA = a.size();
    int lB = b.size();
    int sA;
    VectorXd c(lA* lB);
    c.setZero();
    //cout << c << endl;
    for (int i = 0; i < lA; i++){
        sA = i*lB;
        c.segment(sA, lB) = a(i)* b ;
        //cout << c << endl;
    }
    return c;

}

MatrixXi get_permu_matrix(int n, int m){

    //vector<int> a;//用来存储每次算法产生的当前组合
    int nPermu = tgamma(n+1)/tgamma(m+1)/tgamma(n-m+1) + 0.5;
    MatrixXi res(nPermu, m);
    RowVectorXi a(m);
    int iRow = 0;

    for(int i=0;i<m;i++)//第一种组合，a[0]=1,a[1]=2,...a[m-1]=m;
    {
        a(i) = i +1;
    }

    if(m>n || m<1) {
        cout << "wrong m,n range:"<<m <<","<<n<< endl;
        return res; //确保n>=m>=1
    }

    // 进位法, index i from i to n-m+i
    // 进位同时，让后面的数递增

    cout << "n permutation: "<< nPermu<< ","<< tgamma(n+1)<< ","<< tgamma(m+1)<< ","<< tgamma(n-m+1)<< endl;

    for (int j = 1; j<= nPermu;j++){

        //cout << "give row value begin "<< j<<  endl;
        res.row(iRow)<< a;
        iRow ++;
        //cout << "give row value end"<< endl;
        a(m-1)++;

        for (int k = 1; k < m; k++){
            if (a(m-k) > n-k+1 ){
                if (m-k ==0 ) {
                    break;
                }
                else{
                    a(m-k-1)++;
                    for (int ii = m-k; ii< m; ii++){
                        a(ii) = a(ii-1)+1;
                    }
                }
            } else{ break;}
        }
    }
    return  res;
}

double get_cross_factor( MatrixXd Phi, MatrixXd phi1, MatrixXd phi2, MatrixXd phi3, int S = 0,int  s1 = 0 ,int s2 = 0 , int s3 = 0){
    // calculate  < Phi| phi1> phi2 > phi3 > 
    // to see the phsical meaning
    
    //VectorXd vRight;
    //vRight = 
    return Phi.col(S).dot(v_kronecker_product(v_kronecker_product(phi1.col(s1), phi2.col(s2)), phi3.col(s3)));
}

void d3_stage2( MatrixXd h1, MatrixXd h2, MatrixXd h3, VectorXd v):
    // construt 3d Hamiltonian matrix, plus potential at diag and eigen
    cout << "creat H begin\n";
    int i,j;
    for (int i1 = 0; i1<PODVR_Nn; i1++ ){
        for (int i2 = 0; i2<PODVR_Nn; i2++ ){
            for (int i3 = 0; i3<PODVR_Nn; i3++ ){
                for (int j1 = 0; j1<PODVR_Nn; j1++ ){
                    for (int j2 = 0; j2<PODVR_Nn; j2++ ){
                        for (int j3 = 0; j3<PODVR_Nn; j3++ ){
                            i = get_index_3d(i1,i2,i3, PODVR_Nn);
                            j = get_index_3d(j1,j2,j3, PODVR_Nn);
                            //cout << "begin of T" << i << " " << j << "\n" ;
                            //cout << "Hij"<< H_PODVR(i,j)<< endl;
                            //cout << "H1_ij" << hCc(i1,j1)<< endl;
                            //cout << "H2_ij" << hCh(i2,j2)<< endl;
                            //cout << "H3_ij" << hCn(i3,j3)<< endl;
                            // Is there any simple methmatical form?
                            H_PODVR(i,j) = h1(i1,j1)*delta(i2,j2)*delta(i3,j3) + 
                                           h2(i2,j2)*delta(i1,j1)*delta(i3,j3) +
                                           h3(i3,j3)*delta(i1,j1)*delta(i2,j2);
                            //cout << "end of T" << i << " " << j << "\n" ;
                            if (i==j){
                                H_PODVR(i,j)+= v[i];
                            }
                        }
                    }
                }
            }
        }
    }

    SelfAdjointEigenSolver<MatrixXd> H_es(H_PODVR);
    E=H_es.eigenvalues();
    wf=H_es.eigenvectors();

    const double e_opt = -132.58062407; // a.u.
    for(int i=0; i!=10;++i) outfile <<setw(20)<<(E(i)- e_opt)*hartreeincm<<endl;
    cout << "Transitions (per cm):" <<endl;
    for(int i=0; i!=10;++i){
        cout <<i<<"   "<<(E(i)-E(0))*hartreeincm<<endl;
    }
    cout << endl;

    // wavefunction
    //cout << "WF of Lowest 5 states:" <<endl;
    //for(int i=0; i!=5;++i) outfile << "State " <<i << ":      " <<wf.col(i).transpose()<<endl;
    return 0;

double get_dse(const double f1, const double f2,const double f3){
    // get 3d vibrational level by solving matrix directly 
    // all data used in a.u. , except the x values in data file in angs
    // I am not familiar to C++ for now, so here I decide not to passing vars frequently,
    // which makes it looks ugly ;(

    // --- vars
    // unit
    const double amau=1822.887427;
    const double hartreeincm=2*109737.32;
    const double bohrinang=0.5291772108; // bohr to angs
    // dvr
    const double xmax= 0.95/bohrinang;
    const double xmin=-0.95/bohrinang;//xmin/max for dvr
    double mn=1.0*amau;
    const int PODVR_Nn = 5;
    const int pri_Nn = 201;

    VectorXd gridn(1);
    VectorXd En(1);
    MatrixXd wfCc(1,1);
    MatrixXd wfCh(1,1);
    MatrixXd wfCn(1,1);
    MatrixXd wfn(1,1);
    // creat Matrix
    MatrixXd hCc(1,1);
    MatrixXd hCh(1,1);
    MatrixXd hCn(1,1);
    VectorXd v_grid_cc(PODVR_Nn);
    VectorXd v_grid_ch(PODVR_Nn);
    VectorXd v_grid_cn(PODVR_Nn);
    MatrixXd H_PODVR(PODVR_Nn*PODVR_Nn*PODVR_Nn, PODVR_Nn*PODVR_Nn*PODVR_Nn);
    MatrixXd wf(1,1);
    MatrixXd T(1,1);
    VectorXd E(1);
    // ---

    // --- file IO
    int nDat = 40;
    ofstream outfile("d3.log",ios::out);
    // get v on cross grid
    //VectorXd vCross(PODVR_Nn*PODVR_Nn*PODVR_Nn) ;
    //VectorXd vCross = get_v_cross("d3_7x7x7.dat", PODVR_Nn*PODVR_Nn*PODVR_Nn);
    VectorXd vCross = get_v_cross("v_cross.dat", PODVR_Nn*PODVR_Nn*PODVR_Nn);
    //VectorXd vCross = get_v_cross("/home/jtyang/an/pot_3d/5x5x5_morph/d3.dat", PODVR_Nn*PODVR_Nn*PODVR_Nn);
    //cout << vCross;

    // CC
    // read
    VectorXd x(nDat);
    VectorXd potn(nDat);
    ifstream f("cc.dat",ios::in);
    for ( int i=0;i<nDat;i++)
    {
        f >> x[i] >> potn[i] ;
        x[i]/=bohrinang; // angs to bohr
        //cout << x[i] << ' '<< potn[i] << endl;
    }
    potn = f1* potn;// change potential with factor
    f.close();

    // make spline
    CubicSpline cSpl;
    cSpl.construct(nDat,x,potn,1.0e31,1.0e31);
    cout << "make spline end\n";

    // dvr
    // here I pass a lambda function
    auto potentialn = [&cSpl](double xx){
        return cSpl.calc(xx);
    };

    sinc_podvr_1d(mn,pri_Nn,PODVR_Nn, xmin, xmax, gridn, En, wfCc, hCc, potentialn, true);
    //cout <<"end of dvr\n";

    // output
    outfile << "<CC>"<<endl;
    outfile << "gridn:"<<endl;
    outfile << gridn<<endl;
    outfile <<"e level (1/cm):"<<endl;
    outfile << (En-En(0)*VectorXd::Ones(PODVR_Nn))*hartreeincm<<endl;
    //outfile << En * hartreeincm <<endl;
    outfile << "wfn:"<<endl;
    outfile<<wfn<<endl;
    //

    // v on grids
    for (int i = 0; i< PODVR_Nn; i++ ){
        v_grid_cc[i]  = cSpl.calc(gridn(i));
    }

    // CH
    // read
    f.open("ch.dat",ios::in);
    for ( int i=0;i<nDat;i++)
    {
        f >> x[i] >> potn[i] ;
        x[i]/=bohrinang; // angs to bohr
        //cout << x[i] << ' '<< potn[i] << endl;
    }
    potn = f2* potn;// change potential with factor
    f.close();

    // make spline
    //CubicSpline cSpl;
    //cout << "make spline begin\n";
    cSpl.construct(nDat,x,potn,1.0e31,1.0e31);
    sinc_podvr_1d(mn,pri_Nn,PODVR_Nn, xmin, xmax, gridn, En, wfCh, hCh, potentialn, true);
    // output
    outfile << "<CH>"<<endl;
    outfile << "gridn:"<<endl;
    outfile << gridn<<endl;
    outfile <<"e level (1/cm):"<<endl;
    outfile << (En-En(0)*VectorXd::Ones(PODVR_Nn))*hartreeincm<<endl;
    //outfile << En* hartreeincm <<endl;
    outfile << "wfn:"<<endl;
    outfile<<wfn<<endl;
    //cout << "make spline end\n";

    // v on grids
    for (int i = 0; i< PODVR_Nn; i++ ){
        v_grid_ch[i]  = cSpl.calc(gridn(i));
    }

    // CN
    // read
    f.open("cn.dat",ios::in);
    for ( int i=0;i<nDat;i++)
    {
        f >> x[i] >> potn[i] ;
        x[i]/=bohrinang; // angs to bohr
        //cout << x[i] << ' '<< potn[i] << endl;
    }
    potn = f3* potn;// change potential with factor
    f.close();

    // make spline
    //CubicSpline cSpl;
    //cout << "make spline begin\n";
    cSpl.construct(nDat,x,potn,1.0e31,1.0e31);
    sinc_podvr_1d(mn,pri_Nn,PODVR_Nn, xmin, xmax, gridn, En, wfCn, hCn, potentialn, true);
    // output
    outfile << "<CN>"<<endl;
    outfile << "gridn:"<<endl;
    outfile << gridn<<endl;
    outfile <<"e level (1/cm):"<<endl;
    outfile << (En-En(0)*VectorXd::Ones(PODVR_Nn))*hartreeincm<<endl;
    //outfile << En* hartreeincm <<endl;
    outfile << "wfn:"<<endl;
    outfile<<wfn<<endl;
    //cout << "make spline end\n";

    // v on grids
    for (int i = 0; i< PODVR_Nn; i++ ){
        v_grid_cn[i]  = cSpl.calc(gridn(i));
    }

    // creat H
    // Calculate <phi_i1|<psi_i2|<phi_i3|H_1+H_2+H_3+V123|psi_j3|psi_j2>|phi_j1>
    // 1, cc , 2 ch, 3 cn
    double fCross = f1*f2*f3;
    cout << "creat H begin\n";
    int i,j;
    for (int i1 = 0; i1<PODVR_Nn; i1++ ){
        for (int i2 = 0; i2<PODVR_Nn; i2++ ){
            for (int i3 = 0; i3<PODVR_Nn; i3++ ){
                for (int j1 = 0; j1<PODVR_Nn; j1++ ){
                    for (int j2 = 0; j2<PODVR_Nn; j2++ ){
                        for (int j3 = 0; j3<PODVR_Nn; j3++ ){
                            i = get_index_3d(i1,i2,i3, PODVR_Nn);
                            j = get_index_3d(j1,j2,j3, PODVR_Nn);
                            //cout << "begin of T" << i << " " << j << "\n" ;
                            //cout << "Hij"<< H_PODVR(i,j)<< endl;
                            //cout << "H1_ij" << hCc(i1,j1)<< endl;
                            //cout << "H2_ij" << hCh(i2,j2)<< endl;
                            //cout << "H3_ij" << hCn(i3,j3)<< endl;
                            // Is there any simple methmatical form?
                            H_PODVR(i,j) = hCc(i1,j1)*delta(i2,j2)*delta(i3,j3) + 
                                           hCh(i2,j2)*delta(i1,j1)*delta(i3,j3) +
                                           hCn(i3,j3)*delta(i1,j1)*delta(i2,j2);
                            //cout << "end of T" << i << " " << j << "\n" ;
                            if (i==j){
                                H_PODVR(i,j)+= vCross[i] - v_grid_cc[i1] -v_grid_ch[i2] - v_grid_cn[i3];
                                //H_PODVR(i,j)+= (vCross[i] - v_grid_cc[i1] -v_grid_ch[i2] - v_grid_cn[i3])* fCross;
                            }
                        }
                    }
                }
            }
        }
    }

    cout << "end of creat H" << endl;
    cout << "v 1d on grids:" << endl;
    cout << v_grid_cc << endl;
    cout << v_grid_ch << endl;
    cout << v_grid_cn << endl;
    cout << "---" << endl;

    wrt_eigen_m(H_PODVR);
    //cout << H_PODVR<< endl;

    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    SelfAdjointEigenSolver<MatrixXd> H_es(H_PODVR);
    E=H_es.eigenvalues();
    wf=H_es.eigenvectors();
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout<<"time = "<<timeuse<<endl;  //输出时间（单位：ｓ）

    //output
    //first 5 level
    outfile <<endl;
    outfile <<endl;
    outfile<< "e 3d:"<<endl;
    const double e_opt = -132.58062407; // a.u.
    for(int i=0; i!=10;++i) outfile <<setw(20)<<(E(i)- e_opt)*hartreeincm<<endl;
    outfile << "Transitions (per cm):" <<endl;
    for(int i=0; i!=10;++i){
        outfile<<i<<"   "<<(E(i)-E(0))*hartreeincm<<endl;
    }
    outfile <<endl;

    // wavefunction
    outfile << "WF of Lowest 5 states:" <<endl;
    for(int i=0; i!=5;++i) outfile << "State " <<i << ":      " <<wf.col(i).transpose()<<endl;

    // close
    outfile.close();


    // get DSE
    double wExpCc=920;
    double wExpCh = 1388;
    double wExpCn = 2267;
    double wCc = (E(1)- E(0)) * hartreeincm;
    double wCh = (E(2)- E(0)) * hartreeincm;
    double wCn = (E(4)- E(0)) * hartreeincm;

    //double dse = square(wExpCc - wCc)+ square(wExpCh - wCh) + square(wExpCn - wCn);

    return square(wExpCc - wCc)+ square(wExpCh - wCh) + square(wExpCn - wCn);
}

void opt_factor(double rangeF = 0.1,  int nGrid = 40){
    // optimize factors of morphing PES by scanning then space of factors

    //double f1,f2,f3;
    double iA = 1.0-1* rangeF;
    double iB = 1.0 + rangeF;
    double iStep = (iB-iA)/nGrid;
    double dse = 0 ;
    iB += iStep/2;//let iB included in loop

    //dse = get_dse(1,1,1);
    ofstream outfile("dse.log",ios::out);
    outfile << "f1,f2,f3,dse"<< endl;
    for (double i = iA; i<= iB; i+= iStep ){
        for (double j = iA; j<= iB; j+= iStep ){
            for (double k = iA; k<= iB; k+= iStep ){
                dse = get_dse(i,j,k);
                outfile << i<< "," <<j << ","<<k<<","<<dse<<endl;
            }
        }
    }
    outfile.close();
    cout << "end of get_des"<< endl;
    return;
}



int main(){
    //opt_factor();
    double dse;
    dse = get_dse(1.0,1.0,1.0);
    cout << dse << endl;
    //dse = get_dse(1.025,0.93,0.98);
    //cout << dse << endl;
    return 0;
}
