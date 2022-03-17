import numpy as np

def fastest_desc(A, b, err):
	x = np.zeros_like(b)
	x_new = np.zeros_like(x)
	error = []
	for it_count in range(1, 10000):

		r = A.dot(x) - b
		tau = ((A.dot(r)).dot(r)) / ((A.dot(r)).dot((A.dot(r))))

		x_new = x - tau * r

		if np.allclose(x, x_new, atol=err):
			break

		x = x_new

		error.append(max(abs(A.dot(x) - b)))

	return x, it_count, error

