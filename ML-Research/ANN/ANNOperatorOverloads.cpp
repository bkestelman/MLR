#include "ANN.h"

std::ostream& operator<<(std::ostream& os, const std::vector<ANN::Matrix_t> matrices) {
	for (auto& matrix : matrices) {
		os << matrix << "\n" << std::endl;
	}
	return os;
}
std::ostream& operator<<(std::ostream& os, const std::vector<ANN::val_t> biases) {
	for (auto& bias : biases) {
		os << bias << ", ";
	}
	os << std::endl;
	return os;
}
std::ostream& operator<<(std::ostream& os, const ANN& ann) {
	os << "ANN\n---\n";
	os << "Weights\n";
	os << ann._weights;
	os << "Biases\n";
	os << ann._biases;
	return os;
}
