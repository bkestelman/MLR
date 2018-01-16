#pragma once
#include "Data/DataReader.h"
#include<string>
#include<fstream>
#include<vector>

class ANN;
class MNISTReader : public DataReader {
public:
	MNISTReader(int dataSize);
	DataReader::Vector_t readData() override;
	DataReader::Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	void prepareBuffer() override;
	void seek(int) override;
	bool test(DataReader::Vector_t) override;
	std::string log() override;
	//const void testAssertions(const ANN&) override;
	//friend std::ostream& operator<<(std::ostream&, MNISTReader) override;

	int _dataStep;
	int _scaleDown;

	DataReader::Vector_t _labelCounts;

private:
	struct MNISTFile {
		std::string filename;
		std::ifstream fstream;
		int headers, magicNumber, items, imageWidth, imageHeight;
		int dataPos; 
	};
	int itemsRead;
	int _bufferSize;
	int _next;

	static const int MNISTFileCount = 4;
	MNISTFile MNISTFiles[MNISTFileCount];
	std::vector<Vector_t> _dataBuffer;
	std::vector<Vector_t> _labelBuffer;
	int _curItem;
	Vector_t _label;

	void setupFilenames();
	void setupHeaderCounts();
	void openStreams();
	void readHeaders();
	void readHeaders(MNISTFile& file);	
	Vector_t readDataFromSource(); /* called by prepareBuffer() */
	Vector_t readLabelFromSource();

	int imageFileToRead();
	int labelFileToRead();

	std::ofstream _debug_log; 
};

