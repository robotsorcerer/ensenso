#ifndef ENSENSO_SAVGOL_H_
#define ENSENSO_SAVGOL_H_

#include <Eigen/Core>

Eigen::MatrixXi vander(const int F);
Eigen::MatrixXf sgdiff(int k, int F, double Fd);
template<typename T>
Eigen::RowVectorXf savgolfilt(std::queue<T> const & x, int k, int F);

#endif
