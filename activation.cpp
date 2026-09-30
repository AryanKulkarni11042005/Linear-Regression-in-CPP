#include <iostream>
#include "sum.h"
#include "activation.h"
#include <algorithm>

float activation(Node node) {
	return std::max(0.0f, node.val);
}