#include <stdio.h>
#include <numeric>
#include <math.h>
#include "omp.h"
#include "QPBO.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <future>
#include <cstdlib>


using namespace std;

typedef int CircleId;
typedef int NodeId;

double d_k_e (CircleId k, NodeId x, NodeId y,
              const vector<vector<int> >& CIRCLE, const vector<double>& ALPHA) {
    int delta_e = (CIRCLE[k][x] == 1 && CIRCLE[k][y] == 1) ? 1 : 0;
    double result = (delta_e - ALPHA[k] * (1 - delta_e));
    return result;
}

int phi_dot_theta_k (CircleId k, NodeId x, NodeId y,
                     const vector<vector <vector<int> > >& PHI,
                     const vector<vector<int> >& THETA) {
    int result = (inner_product(PHI[x][y].begin(), PHI[x][y].end(),
                          THETA[k].begin(), 0));
    return result;
}

double o_k_e (CircleId k, NodeId x, NodeId y, int K,
              const vector<vector<int> >& CIRCLE, const vector<double>& ALPHA,
              const vector<vector <vector<int> > >& PHI,
              const vector<vector<int> >& THETA) {
    double result = 0;
    #pragma omp parallel for
    for (CircleId i = 0; i < K; i++)
    {
        if (i != k) {
            result += d_k_e(i, x, y, CIRCLE, ALPHA) *
                        phi_dot_theta_k(i, x, y, PHI, THETA);
        }
    }
    return result;
}

tuple<double, double, double> negPairWiseE (CircleId k, NodeId x, NodeId y, int K,
                     const vector<vector<int> >& EDGE,
                     const vector<vector<int> >& CIRCLE,
                     const vector<double>& ALPHA,
                     const vector<vector <vector<int> > >& PHI,
                     const vector<vector<int> >& THETA) {
    double result[3];
    result[0] = o_k_e(k, x, y, K, CIRCLE, ALPHA, PHI, THETA);
    result[1] = phi_dot_theta_k(k, x, y, PHI, THETA);
    result[2] = result[0] + result[1];
    if (EDGE[x][y] == 1)
    {
        result[2] -= log(1 + exp(result[2]));
    }
    else {
        result[2] = -log(1 + exp(result[2]));
    }
    result[2] = -result[2];
    return make_tuple(result[0], result[1], result[2]);
    // if (dx == 1 && dy == 1)
    // {
    //     result += phi_dot_theta_k(k, x, y, PHI, THETA);
    // }
    // else {
    //     result -= ALPHA[k] * phi_dot_theta_k(k, x, y, PHI, THETA);
    // }
    // if (EDGE[x][y] == 1)
    // {
    //     result -= log(1 + exp(result));
    // }
    // else {
    //     result = -log(1 + exp(result));
    // }
    // return (-result);
}



int main(int argc, char const *argv[])
{
    omp_set_num_threads(2);
    int K = atoi(argv[3]);

    typedef double REAL;
    QPBO<REAL>* q;
    vector<vector<int> > EDGE;
    vector<vector <vector<int> > > PHI;

    ifstream edgeTXT(argv[1]);
    string line;
    int i, M = 0;
    while(getline(edgeTXT, line)) {
        stringstream ss(line);
        vector<int> v;
        while (ss >> i) {
            v.push_back(i);
            if (i == 1) M ++;
        }
        EDGE.push_back(v);
    }
    edgeTXT.close();

    int N = EDGE.size();

    ifstream phiTXT(argv[2]);
    int count = 0;
    int F;
    vector < vector<int> > phi;
    while(getline(phiTXT, line)) {
        stringstream ss(line);
        vector<int> v;
        while (ss >> i) {
            v.push_back(i);
        }
        F = v.size();
        phi.push_back(v);
        count ++;
        if (count % N == 0) {
            PHI.push_back(phi);
            phi.clear();
        }
    }
    phiTXT.close();

    cout << "Nodes: " << N << " Edges: " << M << " Features: " << F << endl;

    ofstream circleTXT("CIRCLE.txt");
    for (CircleId k = 0; k < K; k ++)
    {
        for (NodeId v = 0; v < N; v++)
        {
            circleTXT << "0 ";
        }
        circleTXT.flush();
        if (k != K - 1) circleTXT << endl;
    }



    ofstream thetaTXT("THETA.txt");
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, 1);
    for (CircleId k = 0; k < K; k++)
    {
        for (int f = 0; f < F; f++)
        {
            thetaTXT << distribution(generator) << " ";
        }
        thetaTXT.flush();
        if (k != K - 1) thetaTXT << endl;
    }

    ofstream alphaTXT("ALPHA.txt");
    for (CircleId k = 0; k < K; k++)
    {
        alphaTXT << 1.0 << " ";
    }



    char c;
    do {
        vector< vector<int> > CIRCLE(K, vector<int>(N));

        vector<double> ALPHA(K, 0);

        vector< vector<int> > THETA(K, vector<int>(F));

        ifstream IcircleTXT("CIRCLE.txt");
        for (CircleId k = 0; k < K; k++)
        {
            for (NodeId v = 0; v < N; v++)
            {
                IcircleTXT >> CIRCLE[k][v];
            }
            char newLine;
            IcircleTXT >> newLine;
        }
        IcircleTXT.close();


        ifstream IthetaTXT("THETA.txt");
        for (CircleId k = 0; k < K; k++)
        {
            for (int f = 0; f < F; f++)
            {
                IthetaTXT >> THETA[k][f];
            }
        }
        IthetaTXT.close();

        ifstream IalphaTXT("ALPHA.txt");
        for (CircleId k = 0; k < K; k++)
        {
            IalphaTXT >> ALPHA[k];
        }
        IalphaTXT.close();

        q = new QPBO<REAL>(K * N, K * M); // max number of nodes & edges
        q->AddNode(K * N); // add nodes

        #pragma omp parallel for collapse(3) shared(q)
            for (CircleId k = 0; k < K; k++)
            {
                for (NodeId x = 0; x < N; x++)
                {
                    for (NodeId y = 0; y < N; y++)
                    {
                        if (x == y) continue;
                        double energy0, energy1, energy2;
                        tie(energy0, energy1, energy2) =
                            negPairWiseE (k, x, y, K,
                                        EDGE, CIRCLE, ALPHA, PHI, THETA);

                        double E_1_1 = energy2;
                        double E_0_0 = energy0 - ALPHA[k] * energy1;
                        if (EDGE[x][y] == 1)
                        {
                            E_0_0 -= log(1 + exp(E_0_0));
                        }
                        else {
                            E_0_0 = -log(1 + exp(E_0_0));
                        }
                        E_0_0 = -E_0_0;
                        // double E_0_1 = negPairWiseE (0, 1, k, x, y, K,
                        //                              EDGE, CIRCLE, ALPHA, PHI, THETA);
                        // double E_1_0 = negPairWiseE (1, 0, k, x, y, K,
                        //                              EDGE, CIRCLE, ALPHA, PHI, THETA);
                        // double E_1_1 = negPairWiseE (1, 1, k, x, y, K,
                        //                              EDGE, CIRCLE, ALPHA, PHI, THETA);
                        #pragma omp critical
                        {
                            q->AddPairwiseTerm(k * N + x, k * N + y,
                                               E_0_0, E_0_0, E_0_0, E_1_1);
                        }
                    }
                }
            }

        q->Solve();
        q->ComputeWeakPersistencies();


        bool change = false;
        ofstream OcircleTXT("CIRCLE.txt");
        for (CircleId k = 0; k < K; k++)
        {
            for (NodeId v = 0; v < N; v++)
            {
                int label_k_v = q->GetLabel(k * N + v);
                OcircleTXT << label_k_v << " ";
                if (label_k_v != CIRCLE[k][v]) change = true;
            }
            OcircleTXT.flush();
            if (k != K - 1) OcircleTXT << endl;
        }

        if (!change) {
            cout << "Convergence!" << endl;
            break;
        }



        cout << "This round completed. Another round? [Y/N]: ";
        cin >> c;
    } while (c == 'Y');



    return 0;
}