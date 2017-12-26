#include "ANN.h"

ANN::val_t ANN::nodeBefore(size_t weightLayer, size_t row, size_t col) {
	return _layers[weightLayer](col);
}
ANN::val_t ANN::nodeAfter(size_t weightLayer, size_t row, size_t col) {
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
size_t ANN::weightLayerAfter(size_t nodeLayer) {
	return nodeLayer;
}
size_t ANN::lastWeightLayer() {
	return _weights.size() - 1;
}
size_t ANN::outputLayer() {
	return _layers.size() - 1;
}