#pragma once

#include <vector>

struct Node {
	float val;
	float weight;
	Node(float x, float y) {
		val = x;
		weight = y;
	}
	void hiddenLayer(float weightedSum) {
		val = weightedSum;
	}
};
float weightedSum(std::vector<Node>& n1);
