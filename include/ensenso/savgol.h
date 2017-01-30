#ifndef ENSENSO_SAVGOL_H_
#define ENSENSO_SAVGOL_H_

#include <Eigen/Core>

Eigen::MatrixXi vander(const int F);
Eigen::MatrixXf sgdiff(int k, int F, double Fd);
Eigen::RowVectorXf savgolfilt(Eigen::VectorXf const & x, Eigen::VectorXf const & x_on, int k, int F, double Fd);

#endif
