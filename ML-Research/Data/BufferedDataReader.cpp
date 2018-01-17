#include "BufferedDataReader.h"

BufferedDataReader::BufferedDataReader(int bufferSize) :
	_bufferSize(bufferSize),
	_next(0),
	_dataBuffer(bufferSize),
	_labelBuffer(bufferSize)
{
	//prepareBuffer(); /* TODO: can't be called before BufferedDataReader is constructed and readFromSource methods defined */
}
void BufferedDataReader::init() {
	prepareBuffer();
}

void BufferedDataReader::prepareBuffer() {
	for(int i = 0; i < _bufferSize; i++) {
		_dataBuffer[i] = readDataFromSource();
		_labelBuffer[i] = readLabelFromSource();
	}
	_next = 0;
}

DataReader::Vector_t BufferedDataReader::readData() { /* TODO: read data and label simultaneously, into struct */
	if(_next >= _bufferSize) prepareBuffer();
	return _dataBuffer[_next];
}
DataReader::Vector_t BufferedDataReader::readLabel() { /* TODO: read data and label simultaneously, into struct */
	if(_next >= _bufferSize) prepareBuffer();
	_label = _labelBuffer[_next++];
	return _label;
}
