import imageio

class VideoReader:
	def __init__(self, file):
		self.vid = imageio.get_reader(file)

	def get_next(self):
		return self.vid.get_next_data()

	def __len__(self):
		return len(self.vid)

	def iter_data(self, function, *args):
		for num, image in enumerate(self.vid.iter_data()):
			print(f'frame number: {num + 1}')
			function(image, *args)

	def get_framerate(self):
		return self.vid.get_meta_data()['fps']


class VideoWriter:
	def __init__(self, file, framerate):
		self.writer = imageio.get_writer(file, fps=30)

	def push(self, img):
		self.writer.append_data(img)