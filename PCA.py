from sklearn.decomposition import FastICA, PCA
import numpy as np


if __name__ == '__main__':
	files_name = np.load('./output/file_name.npy')
	print(files_name)

	for num in range(1, 2):
	# for num in range(7, 9):
		i = files_name[num]
		output = np.load('./output/{0}_check_values.npy'.format(i))
		print('h layer', i, output.shape)
		output = output.reshape((60000, -1))
		# output = output.reshape(-1, 1)
		print(output.shape)

		projector = PCA(n_components=5000)
		reduced_activations = projector.fit_transform(output)
		print(reduced_activations.shape)
		print(type(reduced_activations))

		file_name = './adv_output/{0}_PCA_values.npy'.format(i)
		np.save(file_name, reduced_activations)

		output_2 = np.load('./adv_output/{0}_check_values.npy'.format(i))
		output_2 = output_2.reshape((100, -1))
		adv_reduced_activations = projector.transform(output_2)
		print(adv_reduced_activations.shape)
		print(type(adv_reduced_activations))

		file_name = './adv_output/{0}_PCA_values.npy'.format(i)
		np.save(file_name, reduced_activations)
