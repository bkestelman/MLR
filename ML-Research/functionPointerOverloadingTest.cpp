#include<iostream>

int main() {
	funcA();
	funcB();
}

void funcA() {
	std::cout << "funcA" << std::endl;
}

void funcB(int x) {
	std::cout << "funcB(" << x << ")" << std::endl;
}