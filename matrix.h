#pragma once

#include <memory>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <stdexcept> 
#include <random>
#include "datatype.h"
//this will have all the matrix functions 
template <typename T>
struct Matrix
{
	std::size_t rows, cols;
	std::unique_ptr<T[]> data;
	
	Matrix(std::size_t r, std::size_t c)
		: rows(r), cols(c), data(std::make_unique<T[]>(r*c)) {
		//std::cout << "Matrix created" << std::endl;
	};

	Matrix(const Matrix& other)
		:rows(other.rows), cols(other.cols), data(std::make_unique<T[]>(other.rows * other.cols))
	{
		std::size_t n = other.rows * other.cols;
		//std::cout << "Copy Created" << std::endl;
		std::copy(other.data.get(), other.data.get() + n, data.get());
	}
	Matrix& operator=(const Matrix& other)
	{
		if (this == &other)
		{
			return *this;
		}
		
		std::size_t n = other.rows * other.cols;
		auto newData = std::make_unique<T[]>(n);
		std::copy(other.data.get(),other.data.get() + n, newData.get());
		data = std::move(newData);
		rows = other.rows;
		cols = other.cols;
		//std::cout << "Copy Assigned" << std::endl;
		return *this;
	}
	Matrix(Matrix&& other) noexcept
		: rows(other.rows), cols(other.cols), data(std::move(other.data))
	{
		other.rows = 0;
		other.cols = 0;
		//std::cout << "Move Created" << std::endl;
	}

	Matrix& operator=(Matrix&& other) noexcept
	{
		if (this == &other)
		{
			return *this;
		}

		data = std::move(other.data);   // our old array is freed here
		rows = other.rows;
		cols = other.cols;

		other.rows = 0;
		other.cols = 0;

		//std::cout << "Move Assigned" << std::endl;
		return *this;
	}
	T& at(std::size_t i, std::size_t j)
	{
		return data[i * cols + j];
	}

	const T& at(std::size_t i, std::size_t j) const
	{
		return data[i * cols + j];
	}
	void print()
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				std::cout << data[i * cols + j] << " ";
			}
			std::cout << std::endl;
		}
	}
};

template <typename T>
Matrix<T> add(const Matrix<T>& a, const Matrix<T>& b)
{
	if (a.rows != b.rows || a.cols != b.cols)
	{
		throw std::invalid_argument("Rows and Columns of matrices don't match");
	}
	Matrix<T> output(a.rows, a.cols);
	for (std::size_t i = 0; i < (a.rows * a.cols); i++)
	{
		output.data[i] = a.data[i] + b.data[i];
	}
	return output;
}
template <typename T>
Matrix<T> transpose(const Matrix<T>& a)
{
	Matrix<T> output(a.cols, a.rows);
	for (std::size_t i = 0; i < output.rows; i++)
	{
		for (std::size_t j = 0; j < output.cols; j++)
		{
			output.at(i, j) = a.at(j, i);
		}
	}
	return output;
}
template <typename T>
Matrix<T> subtract(const Matrix<T>& a, const Matrix<T>& b)
{
	if (a.rows != b.rows || a.cols != b.cols)
	{
		throw std::invalid_argument("Rows and Columns of matrices don't match");
	}
	Matrix<T> output(a.rows, a.cols);
	for (std::size_t i = 0; i < (a.rows * a.cols); i++)
	{
		output.data[i] = a.data[i] - b.data[i];
	}
	return output;
}

template <typename T>
Matrix<T> scale(const Matrix<T>& a, T s)
{
	Matrix<T> output(a.rows, a.cols);
	for (std::size_t i = 0; i < (a.rows * a.cols); i++)
	{
		output.data[i] = a.data[i] * s;
	}
	return output;
}

template <typename T>
Matrix<T> mat_mul(const Matrix<T>& a, const Matrix<T>& b)
{
	if (a.cols != b.rows)
	{
		throw std::invalid_argument("Matrix Multiplication not possible");
	}
	Matrix<T> output(a.rows, b.cols);
	for (std::size_t i = 0; i < a.rows; i++)
	{
		for (std::size_t j = 0; j < b.cols; j++)
		{
			T sum = 0;
			for (std::size_t p = 0; p < a.cols; p++)
			{
				sum += a.at(i, p) * b.at(p, j);
			}
			output.at(i, j) = sum;
		}
	}
	return output;
}
template <typename T>
T mse(const Matrix<T>& pred, const Matrix<T>& target)
{
	if (pred.rows != target.rows || pred.cols != target.cols)
	{
		throw std::invalid_argument("Rows or Columns of Target and Prediction Don't Match");
	}
	T sum = 0;
	for (std::size_t i = 0; i < pred.rows * pred.cols; i++)
	{
		T diff = pred.data[i] - target.data[i];
		sum += diff * diff;
	}
	return sum / static_cast<T>(pred.rows * pred.cols);
}



void LinearRegression();
void matrixLesson();