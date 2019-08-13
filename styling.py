import argparse
import os

from keras.preprocessing.image import save_img

from utils.Styler import style
from utils.video_utils import VideoReader, VideoWriter


def parse():
	parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')

	parser.add_argument('--mode', type=str, default='img', choices=['img', 'vid'],
	                    help='Specify if u want to transform an image or a video, (default: %(default)s)')
	parser.add_argument('--base_image', type=str, help='Path to the image to transform.')
	parser.add_argument('--base_video', type=str, help='Path to the video to transform.')
	parser.add_argument('--style_image', type=str, help='Path to the style reference image.', required=True)

	parser.add_argument('--output_name', type=str, help='Specify the name and the path for the output image or video',
	                    required=True)

	parser.add_argument('--output_size', type=int, default=512,
	                    help='Specify the height of the output image, default value 512.', required=False)
	parser.add_argument('--fun_evals', type=int, default=100,
	                    help='Specify the number of iterations to apply the styling. This value pretty much represents the time needed to style an image, the higher the value the longer it takes. (default: %(default)s)',
	                    required=False)

	parser.add_argument('--weights', type=str, default='imagenet',
	                    help='Path to a .h5 weight file, containing weights for a VGG19 model, if not set the imagenet weights will be loaded.',
	                    required=False)
	parser.add_argument('--pooling', type=str, default='max', choices=['avg', 'max'],
	                    help='Change the pooling layers. (default: %(default)s)',
	                    required=False)

	parser.add_argument('--tv_weight', type=float, default=1e-3,
	                    help='Specify the total variation weight. (default: %(default)s)',
	                    required=False)
	parser.add_argument('--style_weight', type=float, default=1e4,
	                    help='Specify the style weight. (default: %(default)s)',
	                    required=False)
	parser.add_argument('--content_weight', type=float, default=5e0,
	                    help='Specify the content weight. (default: %(default)s)',
	                    required=False)

	args = parser.parse_args()

	return args

def make_dir(path):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(out_path))

def main():
	global args
	args = parse()

	if args.mode == 'img':
		# create the output directories
		out_path = args.output_name + '.png'
		make_dir(out_path)


		s = style(base_img=args.base_image, style=args.style_image, size=args.output_size, fun_evals=args.fun_evals,
		          pooling=args.pooling, weights=args.weights, tv_weight=args.tv_weight, style_weight=args.style_weight,
		          content_weight=args.content_weight)

		# get the generated image
		oimg = s(args.base_image)

		# save the image
		save_img(out_path, oimg)

	elif args.mode == 'vid':
		v_reader = VideoReader(file=args.base_video)

		# create the output directories
		out_path = args.output_name + os.path.splitext(args.base_video)[1]
		make_dir(out_path)

		v_writer = VideoWriter(file=out_path, framerate=v_reader.get_framerate())

		# get 1 frame of the video for tensor generation
		img = v_reader.get_next()

		s = style(base_img=img, style=args.style_image, size=args.output_size, fun_evals=args.fun_evals,
		          pooling=args.pooling, weights=args.weights, tv_weight=args.tv_weight, style_weight=args.style_weight,
		          content_weight=args.content_weight)

		v_reader.iter_data(function=lambda img: v_writer.push(s(img)))


if __name__ == '__main__':
	main()
