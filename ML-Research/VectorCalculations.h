#pragma once
#include<Eigen\Dense>

int maxIndex(Eigen::VectorXd vec) {
	assert(vec.size() > 0);
	int max = vec[0];
	int ret = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > max) {
			max = vec[i];
			ret = i;
		}
	}
	return ret;
}