from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D


def build_vgg(input_tensor=None, weights=None):
	if input_tensor is not None:
		input = Input(tensor=input_tensor, name='input')
	else:
		input = Input(shape=(None, None, 3), name='input')

	# Block 1
	conv = Conv2D(64, kernel_size=3, padding='same', activation='relu', name='block1_conv1')(input)
	conv = Conv2D(64, kernel_size=3, padding='same', activation='relu', name='block1_conv2')(conv)
	pooling = AveragePooling2D(padding='same', name='block1_pooling')(conv)

	# Block 2
	conv = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='block2_conv1')(pooling)
	conv = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='block2_conv2')(conv)
	pooling = AveragePooling2D(padding='same', name='block2_pooling')(conv)

	# Block 3
	conv = Conv2D(256, kernel_size=3, padding='same', activation='relu', name='block3_conv1')(pooling)
	conv = Conv2D(256, kernel_size=3, padding='same', activation='relu', name='block3_conv2')(conv)
	conv = Conv2D(256, kernel_size=3, padding='same', activation='relu', name='block3_conv3')(conv)
	conv = Conv2D(256, kernel_size=3, padding='same', activation='relu', name='block3_conv4')(conv)
	pooling = AveragePooling2D(padding='same', name='block3_pooling')(conv)

	# Block 4
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block4_conv1')(pooling)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block4_conv2')(conv)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block4_conv3')(conv)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block4_conv4')(conv)
	pooling = AveragePooling2D(padding='same', name='block4_pooling')(conv)

	# Block 5
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block5_conv1')(pooling)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block5_conv2')(conv)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block5_conv3')(conv)
	conv = Conv2D(512, kernel_size=3, padding='same', activation='relu', name='block5_conv4')(conv)
	pooling = AveragePooling2D(padding='same', name='block5_pooling')(conv)

	model = Model(input, pooling)

	if weights:
		model.load_weights(weights)

	return Model(input, pooling)