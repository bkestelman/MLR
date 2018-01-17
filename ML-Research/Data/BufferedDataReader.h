#pragma once
#include "DataReader.h"
#include<vector>

class BufferedDataReader : public DataReader {
	//~BufferedDataReader() {};
protected:
	int _bufferSize, _next;
	std::vector<Vector_t> _dataBuffer, _labelBuffer;

	BufferedDataReader(int bufferSize);
	void prepareBuffer();
	virtual Vector_t readDataFromSource() = 0;
	virtual Vector_t readLabelFromSource() = 0;	
	Vector_t _label;
public:
	Vector_t readData() override;
	Vector_t readLabel() override;
	void init() override;
};	
