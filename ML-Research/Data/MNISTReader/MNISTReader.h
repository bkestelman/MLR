#pragma once
#include "Data/BufferedDataReader.h"
#include<string>
#include<fstream>
#include<vector>

class ANN; /* TODO: delete */
class MNISTReader : public BufferedDataReader {
public:
	MNISTReader(int dataSize);
	const size_t dataSize() override;
	const size_t labelSize() override;
	void seek(int) override;
	bool test(DataReader::Vector_t) override;
	std::string log() override;
	//const void testAssertions(const ANN&) override;
	//friend std::ostream& operator<<(std::ostream&, MNISTReader) override;

	int _dataStep;
	int _scaleDown;

	DataReader::Vector_t _labelCounts;

protected:
	Vector_t readDataFromSource() override; /* called by prepareBuffer() */
	Vector_t readLabelFromSource() override;
private:
	struct MNISTFile {
		std::string filename;
		std::ifstream fstream;
		int headers, magicNumber, items, imageWidth, imageHeight;
		int dataPos; 
	};
	int itemsRead;

	static const int MNISTFileCount = 4;
	MNISTFile MNISTFiles[MNISTFileCount];
	int _curItem;

	void setupFilenames();
	void setupHeaderCounts();
	void openStreams();
	void prepareBuffer();
	void readHeaders();
	void readHeaders(MNISTFile& file);	

	int imageFileToRead();
	int labelFileToRead();

	std::ofstream _debug_log; 
};

