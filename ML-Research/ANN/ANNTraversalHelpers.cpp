#include "ANN.h"

ANN::val_t ANN::nodeValBefore(size_t weightLayer, size_t row, size_t col) {
	return _layers[weightLayer](col);
}
ANN::val_t ANN::nodeValAfter(size_t weightLayer, size_t row, size_t col) {
	return _layers[weightLayer + 1](row);
}
size_t ANN::nodeLayerAfter(size_t weightLayer) {
	return weightLayer + 1;
}
size_t ANN::nodeLayerBefore(size_t weightLayer) {
	return weightLayer;
}
ANN::val_t ANN::weight(size_t weightLayer, size_t nodeBefore, size_t nodeAfter) {
	return _weights[weightLayer](nodeAfter, nodeBefore);
}
ANN::val_t& ANN::weightDelta(size_t weightLayer, size_t nodeBefore, size_t nodeAfter) {
	return _weightsDeltas[weightLayer](nodeAfter, nodeBefore);
}
size_t ANN::weightLayerAfter(size_t nodeLayer) const {
	return nodeLayer;
}
size_t ANN::weightLayerBefore(size_t nodeLayer) {
	return nodeLayer-1;
}
std::size_t ANN::lastWeightLayer() {
	return _weights.size() - 1;
}
size_t ANN::outputLayer() {
	return _layers.size() - 1;
}

