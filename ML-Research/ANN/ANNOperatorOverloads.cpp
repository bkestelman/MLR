#include "ANN.h"

std::ostream& operator<<(std::ostream& os, const std::vector<ANN::Matrix_t> matrices) {
	for (auto& matrix : matrices) {
		os << matrix << "\n" << std::endl;
	}
	return os;
}
std::ostream& operator<<(std::ostream& os, const std::vector<ANN::Vector_t> biases) {
	for (auto& bias : biases) {
		os << bias << "\n\n";
	}
	os << std::endl;
	return os;
}
std::ostream& operator<<(std::ostream& os, const ANN& ann) { /* TODO: */
	os << "ANN\n---\n";
	os << "Weights\n";
	os << ann._weights;
	os << "Layers\n";
	os << ann._layers;
	return os;
}
