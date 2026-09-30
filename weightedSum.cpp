#include <iostream>
#include "sum.h"


float weightedSum(std::vector<Node>& nodes) {
	float sum = 0;
	int n = nodes.size();
	float bias = 1;
	for (int i = 0; i < n; i++) {
		sum += (nodes[i].val * nodes[i].weight);
	}
	return sum;
}