#include <stdio.h>
#include <numeric>
#include <math.h>
#include "QPBO.h"

using namespace std;

typedef int CircleId;
#define N node_num;
#define M edge_num;
#define F feature_size;
#define K circle_num;

double d_k_e (CircleId k, NodeId x, NodeId y,
              int[][] CIRCLE, double[] ALPHA) {
    int delta_e = (CIRCLES[k][x] == 1 && CIRCLES[k][y] == 1) ? 1 : 0;
    return (delta_e - ALPHA[k] * (1 - delta_e));
}

int phi_dot_theta_k (CircleId k, NodeId x, NodeId y,
                     int[][][] PHI, int[][] THETA) {
    return (std::inner_product(PHI[x][y], PHI[x][y] + F, THETA[k], 0));
}

double o_k_e (CircleId k, NodeId x, NodeId y,
              int[][] CIRCLE, double[] ALPHA, int[][][] PHI, int[][] THETA) {
    double result = 0;
    for (CircleId i = 0; i < K; ++i)
    {
        if (i != k) {
            result += d_k_e(i, x, y, CIRCLE, ALPHA) *
                        phi_dot_theta_k(i, x, y, PHI, THETA);
        }
    }
}

double negPairWiseE (int dx, int dy, CircleId k, NodeId x, NodeId y,
                     int[][] EDGE, int[][] CIRCLE, double[] ALPHA,
                     int[][][] PHI, int[][] THETA) {
    double result = o_k_e(k, x, y, CIRCLE, ALPHA, PHI, THETA);
    if (dx == 1 && dy == 1)
    {
        result += phi_dot_theta_k(k, x, y, PHI, THETA);
    }
    else {
        result -= ALPHA[k] * phi_dot_theta_k(k, x, y, PHI, THETA);
    }
    if (EDGE[x][y] == 1)
    {
        result -= log(1 + exp(result));
    }
    else {
        result = -log(1 + exp(result));
    }
    return (-result);
}


int main()
{
    typedef double REAL;
    QPBO<REAL>* q;

    q = new QPBO<REAL>(K * N, K * M); // max number of nodes & edges
    q->AddNode(N); // add two nodes

    for (CircleId k = 0; k < K; ++k)
    {
        for (NodeId x = 0; x < N; ++x)
        {
            for (NodeId y = 0; y < N; ++y)
            {
                double E_0_0 = negPairWiseE (0, 0, k, x, y,
                                             EDGE, CIRCLE, ALPHA, PHI, THETA);
                double E_0_1 = negPairWiseE (0, 1, k, x, y,
                                             EDGE, CIRCLE, ALPHA, PHI, THETA);
                double E_1_0 = negPairWiseE (1, 0, k, x, y,
                                             EDGE, CIRCLE, ALPHA, PHI, THETA);
                double E_1_1 = negPairWiseE (1, 1, k, x, y,
                                             EDGE, CIRCLE, ALPHA, PHI, THETA);
                q->AddPairwiseTerm(k * N + x, k * N + y,
                                   E_0_0, E_0_1, E_1_0, E_1_1);
            }
        }
    }

    q->Solve();
    q->ComputeWeakPersistencies();

    for (CircleId k = 0; k < K; ++k)
    {
        for (NodeId i = 0; i < N; ++i)
        {
            printf("%d ", q->GetLabel(k * N + i));
        }
        printf("\n");
    }

    return 0;
}