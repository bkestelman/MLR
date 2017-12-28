#include "SystemInfo.h"

bool SystemInfo::bigEndian = SystemInfo::testBigEndian();

bool SystemInfo::testBigEndian() {
	int x = 1;
	char* c = (char*)&x;
	if (c[0] == 1) return false; /* c[0] is LSB */
	else return true; /* c[0] is MSB */
}
