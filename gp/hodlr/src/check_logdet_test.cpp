#include <cmath>
#include <ctime>
#include <vector>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>

#include "HODLR_Tree.hpp"
#include "HODLR_Matrix.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;


class Gaussian_Matrix : public HODLR_Matrix {

public:
    Gaussian_Matrix (double a, double s, VectorXd time)
        : amp_(a*a), ivar_(1.0/(s*s)), time_(time) {};

    double get_Matrix_Entry(const unsigned i, const unsigned j) {
        double d = time_[i] - time_[j];
        return amp_ * exp(-0.5 * d * d * ivar_);
    }

private:

    double amp_, ivar_;
    VectorXd time_;

};

int main ()
{
    // Build the times and diagonal arrays.
    int N = 50;
    VectorXd time(N), diag(N);

    for (int i = 0; i < N; ++i) {
        time[i] = i / 24. / 60.;
        diag[i] = 1.0;
    }

    // Set up the solver.
    Gaussian_Matrix matrix(0.999, 5.0, time);
    HODLR_Tree<Gaussian_Matrix>* A =
        new HODLR_Tree<Gaussian_Matrix>(&matrix, N, 50);

    // Assemble/factorize.
    A->assemble_Matrix(diag, 1e-12);
    A->compute_Factor();

    // Compute the determinant.
    double logdet;
    A->compute_Determinant(logdet);

    cout << std::setprecision(16) << logdet << endl;

    delete A;
    return 0;
}
