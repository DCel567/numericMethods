import numpy as np


def zadel(A, b, err):
	x = np.zeros_like(b)
	error = []
	for it_count in range(1, 10000):
		x_new = np.zeros_like(x)

		for i in range(A.shape[0]):
			s1 = np.dot(A[i, :i], x_new[:i])
			s2 = np.dot(A[i, i + 1:], x[i + 1:])
			x_new[i] = (b[i] - s1 - s2) / A[i, i]

		if np.allclose(x, x_new, atol=err):
			break

		x = x_new

		error.append(max(abs(A.dot(x) - b)))

	return x, it_count, error
