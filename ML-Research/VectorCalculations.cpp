#include "VectorCalculations.h"

int maxIndex(Eigen::VectorXd vec) {
	assert(vec.size() > 0);
	double max = vec[0];
	int ret = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > max) {
			max = vec[i];
			ret = i;
		}
	}
	return ret;
}