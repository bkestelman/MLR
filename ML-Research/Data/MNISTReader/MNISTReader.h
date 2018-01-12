#pragma once
#include "Data/DataReader.h"
#include<string>
#include<fstream>
#include<vector>

class ANN;
class MNISTReader : public DataReader {
public:
	MNISTReader();
	DataReader::Vector_t readData() override;
	DataReader::Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
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
	};
	static const int MNISTFileCount = 4;
	MNISTFile MNISTFiles[MNISTFileCount];
	std::vector<int> _dataBuffer;
	int _curItem;
	Vector_t _label;

	void setupFilenames();
	void setupHeaderCounts();
	void openStreams();
	void readHeaders();
	void readHeaders(MNISTFile& file);	
	//void readDataToBuffer();

	int imageFileToRead();
	int labelFileToRead();

	int itemsRead;

	std::ofstream _debug_log; 
};

