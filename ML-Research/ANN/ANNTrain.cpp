#include "ANN.h"
#include<numeric>

void ANN::train() {
	_trains++;
	readNext();
	setInput();
	val_t errBefore = error();
	backprop(); 
	val_t errAfter = error();
	_trainLog << "err before: " << errBefore << "\nerr after: " << errAfter << "\n";
}
void ANN::train(int epochs) { /* DEPRECATED */
	for(int i = 0; i < epochs; i++) {
		std::cout << "i: " << i << "\n";
		train();
	}
}
void ANN::trainBatch() {
	_dr.seek(0);
	for(int i = 0; i < _params._batchSize; i++) {
		std::cout << "i: " << i << "\n";
		train();
	}
}
void ANN::trainBatch(int epochs) {
	for(int i = 0; i < epochs; i++) {
		trainBatch();
	}
}

void ANN::backprop() {
	Vector_t bp_delta(_layerSizes[outputLayer()]);
	Vector_t output_delta = Vector_t::Zero(_layerSizes[outputLayer()]);
	output_delta = processInput() - _label; /* (Y(2) - d)_j */

	Vector_t I = weightedSum(outputLayer() - 1); /* I(2)_j */
	for (int j = 0; j < _layerSizes[outputLayer()]; j++) { /* j indexes outputs (exclude bias node) */
		bp_delta[j] = output_delta[j] * _params.activationFunc_d(I(j)); /* (d - Y(2))_j g'(I(2)_j)*/
	}

	for (auto weightLayer = lastWeightLayer(); weightLayer > 0; weightLayer--) { /* nasty unsigned trick here */
		adjustWeights(weightLayer, bp_delta);
		bp_delta = backpropLayer(weightLayer, bp_delta);
	}
	adjustWeights(0, bp_delta);
	//commitAdjustments();
}

ANN::Vector_t ANN::backpropLayer(size_t weightLayer, Vector_t& bp_delta) { /* TODO: check for memory leak */	
	//commitAdjustments(weightLayer);
	Vector_t new_delta = backprop_delta(weightLayer, bp_delta);	/* TODO: test before, test after */
	return new_delta; /* pass this to the next backPropLayer call */
}

void ANN::adjustWeights(size_t weightLayer, Vector_t& bp_delta) {
	assert(_weights[weightLayer].rows() == sizeNoBias(nodeLayerAfter(weightLayer)));
	assert(_weights[weightLayer].cols() == sizeWithBias(nodeLayerBefore(weightLayer)));
	for (int i = 0; i < _layers[nodeLayerBefore(weightLayer)].size(); i++) { 
		for (int j = 0; j < _layerSizes[nodeLayerAfter(weightLayer)]; j++) { 
			/* W(2)ij = W(2)ij + eta delta(2)_j Y(1)_i */
			_weights[weightLayer](j, i) -= _params._learningRate * bp_delta(j) * _layers[nodeLayerBefore(weightLayer)](i);
		}
	}
}

ANN::Vector_t ANN::backprop_delta(size_t weightLayer, Vector_t& bp_delta) { 
	size_t beforeWeights = nodeLayerBefore(weightLayer); 
	size_t afterWeights = beforeWeights + 1; 
	Vector_t new_delta = Vector_t::Zero(_layers[beforeWeights].size()); 
	Vector_t I = weightedSum(nodeLayerBefore(weightLayer) - 1);
	for (int j = 0; j < _layerSizes[beforeWeights]; j++) { /* bias node also gets a bp_delta */
		for (int k = 0; k < _layerSizes[afterWeights]; k++) { /* but no bp_delta obtained from bias node (no weights to bias node) */
			new_delta[j] += bp_delta[k] * _weights[weightLayer](k, j) * _params.activationFunc_d(I[j]);
			/* delta(1)_i = -sum(delta(2)_j * W(2)_ij) g'(I(1)_i) */
		}
	}

	return new_delta;
}

void ANN::commitAdjustments() { /* TODO: commit each layer as you backprop? */
	for (int i = 0; i < _weights.size(); i++) {
		_weights[i] += _weightsDeltas[i]; 
		_weightsDeltas[i] = Matrix_t::Zero(_weightsDeltas[i].rows(), _weightsDeltas[i].cols());
	}
}
void ANN::commitAdjustments(size_t weightLayer) { /* TODO: commit each layer as you backprop? */
	_weights[weightLayer] += _weightsDeltas[weightLayer];
	_weightsDeltas[weightLayer] = Matrix_t::Zero(_weightsDeltas[weightLayer].rows(), _weightsDeltas[weightLayer].cols());
}
