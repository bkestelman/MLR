#include "ANN.h"

ANN::Vector_t& ANN::train() {
	scale(_input);
	backprop(processInput(), _label);
	return _layers[outputLayer()];
}

/*
ANN::Vector_t& ANN::train(const Vector_t& input, const Vector_t& label) {
	//assert(_layers[outputLayer()].size() == label.size());
	backprop(processInput(), label);
	return _layers[outputLayer()];
}
*/

void ANN::backprop(const Vector_t& output, const Vector_t& label) {
	Vector_t dE_dn = output_dE_dn(output, label); /* Error function derivatives for each output node */
	for (auto weightLayer = lastWeightLayer(); weightLayer+1 > 0; weightLayer--) {
		dE_dn = backpropLayer(weightLayer, dE_dn);                   
	}
}

/* dE_dn is the backpropagated derivative of error function with respect to nodes in the node layer after weightLayer */
ANN::Vector_t ANN::backpropLayer(size_t weightLayer, const Vector_t& backpropagated_dE_dn) { /* TODO: check for memory leak */	
	Vector_t new_dE_dn = backprop_dE_dn(weightLayer, backpropagated_dE_dn);	
	adjustWeights(weightLayer, backpropagated_dE_dn); 
	adjustBiases(weightLayer, backpropagated_dE_dn);
	return new_dE_dn; /* pass this to the next backPropLayer call */
}

void ANN::adjustWeights(size_t weightLayer, const Vector_t& backpropagated_dE_dn) {
	Vector_t n = _layers[nodeLayerAfter(weightLayer)];
	Matrix_t deltas = Matrix_t::Zero(_weights[weightLayer].rows(), _weights[weightLayer].cols());
	val_t totalDelta = 0;
	for (auto row = 0; row < _weights[weightLayer].rows(); row++) {
		val_t rowDelta = 0;
		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
			val_t dn_dw = dn_dw_f(n, row, nodeValBefore(weightLayer, row, col));
			val_t dE_dw = dn_dw * backpropagated_dE_dn[row]; /* chain rule */
			deltas(row, col) = dE_dw;
			//assert(deltas(row, col) != deltas(0, 0));
			rowDelta += dE_dw;
		}
		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
			if (rowDelta != 0) _weights[weightLayer](row, col) -= deltas(row, col) * _stepFactor;// / abs(rowDelta);
		}
	}
	/*
	for (auto row = 0; row < _weights[weightLayer].rows(); row++) {
		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
			_weights[weightLayer](row, col) -= deltas(row, col) * _stepFactor;
		}
	}
	*/
}

void ANN::adjustBiases(size_t weightLayer, const Vector_t& backpropagated_dE_dn) {
	return;
	Vector_t n = _layers[nodeLayerAfter(weightLayer)];
	val_t dE_db = 0;
	for (auto node = 0; node < _layers[nodeLayerAfter(weightLayer)].size(); node++) {
		val_t dn_db = dn_db_f(n, node);
		dE_db += dn_db * backpropagated_dE_dn[node];
	}
	_biases[weightLayer] -= dE_db * _stepFactor;
}

ANN::Vector_t ANN::backprop_dE_dn(size_t weightLayer, const Vector_t& backpropagated_dE_dn) {
	size_t nodeLayerBeforeWeights = nodeLayerBefore(weightLayer);
	size_t nodeLayerAfterWeights = nodeLayerBeforeWeights + 1;
	Vector_t new_dE_dn{ _layers[nodeLayerBeforeWeights].size() };
	Vector_t n = _layers[nodeLayerAfterWeights];
	for (auto node = 0; node < _layers[nodeLayerBeforeWeights].size(); node++) {
		val_t dE_dn = 0; 
		for (auto nextLayerNode = 0; nextLayerNode < _layers[nodeLayerAfterWeights].size(); nextLayerNode++) {
			val_t expo = std::exp(-n[nextLayerNode]);
			val_t dN_dn = dN_dn_f(n, nextLayerNode, weight(weightLayer, node, nextLayerNode)); // TODO: fp
			dE_dn += dN_dn * backpropagated_dE_dn[nextLayerNode]; /* chain rule */
		}
		new_dE_dn[node] = dE_dn;
	}
	return new_dE_dn;
}