import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization


class ResBlock(Layer):

	def __init__(
		self,
		dilation_rate,
		nb_filters,
		kernel_size, 
		padding,
		dropout_rate=0.0,
		kernel_initializer: str = 'he_normal',
		activation: str = 'relu',
		use_batch_norm: bool = False,
		use_layer_norm: bool = False,
		**kwargs
	):

		self.dilation_rate = dilation_rate
		self.nb_filters = nb_filters
		self.kernel_size = kernel_size
		self.padding = padding
		self.activation = activation
		self.dropout_rate = dropout_rate
		self.use_batch_norm = use_batch_norm
		self.use_layer_norm = use_layer_norm
		self.kernel_initializer = kernel_initializer
		
		self.layers = []          # record all layers
		self.final_output_shape = None  # record output shape

		super(ResBlock, self).__init__(**kwargs)


	def build(self, input_shape):

		self.layers = []
		self.final_output_shape = input_shape
		for i in range(2):  # two fcn blocks
			fcn = Conv1D(
				filters=self.nb_filters,
				kernel_size=self.kernel_size,
				dilation_rate=self.dilation_rate,
				padding=self.padding,
				kernel_initializer=self.kernel_initializer
			)
			self.layers.append(fcn)
			self.layers[-1].build(self.final_output_shape)
			self.final_output_shape = fcn.compute_output_shape(self.final_output_shape)  # update output shape

			if self.use_batch_norm:
				self.layers.append(BatchNormalization())
				self.layers[-1].build(self.final_output_shape)
			elif self.use_layer_norm:
				self.layers.append(LayerNormalization())
				self.layers[-1].build(self.final_output_shape)
			
			self.layers.append(Activation(self.activation))
			self.layers[-1].build(self.final_output_shape)
			self.layers.append(SpatialDropout1D(rate=self.dropout_rate))
			self.layers[-1].build(self.final_output_shape)

		if self.nb_filters != input_shape[-1]:
			self.conv1d = Conv1D(
				filters=self.nb_filters,
				kernel_size=1,
				padding='same',
				kernel_initializer=self.kernel_initializer
			)  # match the shape of input and output
		else:
			self.conv1d = Lambda(lambda x: x)
		self.conv1d.build(input_shape)
		self.activation = Activation(self.activation)
		self.activation.build(self.final_output_shape)

		super(ResBlock, self).build(input_shape)

	def compute_output_shape(self, input_shape):

		return [self.final_output_shape, self.final_output_shape]

	def call(self, inputs, training):

		x = inputs
		for layer in self.layers:
			x = layer(x, training=training)  # fcn output
		x1 = self.conv1d(inputs)
		x2 = layers.add([x, x1])
		x3 = self.activation(x2)  # skip connection output after activation

		return [x3, x]


class TempConvNet(Layer):

	def __init__(
		self,
		nb_filters=32,
		kernel_size=3,
		nb_stacks=1,
		dilations=(1, 2, 4, 8, 16),
		padding='causal',
		use_skip_connections=True,
		dropout_rate=0.0,
		return_sequences=False,
		activation='relu',
		kernel_initializer='he_normal',
		use_batch_norm=False,
		use_layer_norm=True,
		**kwargs):

		self.nb_filters = nb_filters
		self.kernel_size = kernel_size
		self.nb_stacks = nb_stacks
		self.dilations = dilations
		self.padding = padding
		self.use_skip_connections = use_skip_connections
		self.dropout_rate = dropout_rate
		self.return_sequences = return_sequences
		self.activation = activation
		self.kernel_initializer = kernel_initializer
		self.use_batch_norm = use_batch_norm
		self.use_layer_norm = use_layer_norm

		self.skip_connections = []
		self.layer = []
		self.final_output_shape = None

		if isinstance(self.nb_filters, list):
			assert len(self.nb_filters) == len(self.dilations)

		super(TempConvNet, self).__init__(**kwargs)

	def build(self, input_shape):

		self.layers = []
		self.final_output_shape = input_shape
		for i, d in enumerate(self.dilations):

			if isinstance(self.nb_filters, list):
				fltr_num = self.nb_filters[i]
			else:
				fltr_num = self.nb_filters
			if (d & (d - 1)) != 0:
				d = 2 ** i
			else:
				pass
			print(d)
			res_block = ResBlock(
				dilation_rate=2 ** i,
				nb_filters=fltr_num,
				kernel_size=self.kernel_size,
				padding=self.padding,
				activation=self.activation,
				dropout_rate=self.dropout_rate,
				kernel_initializer=self.kernel_initializer,
				use_batch_norm=self.use_batch_norm,
				use_layer_norm=self.use_layer_norm
			)
			self.layers.append(res_block)
			self.layers[-1].build(self.final_output_shape)
			self.final_output_shape = self.layers[-1].final_output_shape

		self.slice = Lambda(lambda x: x[:, -1, :])
		self.slice.build(self.final_output_shape)

		super(TempConvNet, self).build(input_shape)

	def call(self, inputs, training):
		res_x = inputs
		self.skip_connections = []
		for layer in self.layers:
			res_x, x = layer(res_x, training=training)
			self.skip_connections.append(x)
		
		if self.use_skip_connections:
			if len(self.skip_connections) != 1:
				res_x = layers.add(self.skip_connections)  # skip connection output
			else:
				pass

		else:
			pass

		if not self.return_sequences:
			res_x = self.slice(res_x)  # 
		else:
			pass
		
		return res_x


if __name__ == '__main__':

	## TempConvNet test:
	use_skip_connections = True
	return_sequences = False
	input_layer = Input(shape=(32, 4))
	res_x = TempConvNet(
		dilations=[1, 2, 4, 8, 16],
		nb_filters=32,
		kernel_size=3,
		padding='causal',
		dropout_rate=0.0,
		use_layer_norm=True
	)(input_layer)
	x = Dense(1)(res_x)
	output_layer = Activation('sigmoid')(x)
	model = Model(input_layer, output_layer)
	model.compile(loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	print(res_x.shape)
