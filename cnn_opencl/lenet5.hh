/*
	2017-02-21: (Liqiang He)  Show case for OpenCL
	2013-07-07
	Junjie Liu
	version 0
	--- --- ---
	in this his class, we implement a specific LeNet5.
	it's only used for test.
*/

#ifndef _lenet5_hh_
#define _lenet5_hh_

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cstdio>
#include <cstdlib>

#include <string>
#include <map>
#include <vector>

#include "../lib/net.hh"
#include "../lib/parameters.hh"

#define dump_dir "./cnn_test_load_dump/"
#define LR_threshold 0.00002


using namespace std;

class mcp_bprop;
class conv_subnet3D;
class pooling_subnet2D;
class neurons;
class neurons2D;
class synapses;
class parameters;

class lenet5 : public net
{
public:
	// neurons
	vector<neurons2D*> input_neurons;
	vector<neurons2D*> conv1_neurons;
	vector<neurons2D*> pooling2_neurons;
	vector<neurons2D*> conv3_neurons;
	vector<neurons2D*> pooling4_neurons;
	vector<neurons2D*> conv5_neurons;
	neurons* hidden6_neurons;
	neurons* output_neurons;

	// synapses
	// the first dimention: output feature maps
	// the second dimention: input feature maps
	vector<vector<synapses*> > input_conv1_synapses;
	vector<vector<synapses*> > pooling2_conv3_synapses;
	vector<vector<synapses*> > pooling4_conv5_synapses;
	synapses* conv5_hidden6_synapse;
	synapses* hidden6_output_synapse;

	// connect tables
	// we need a connect table only in convolutional layers(include input layer, when the input layer has several input featuremaps)
	vector<vector<bool> > pooling2_conv3_connection;
	vector<vector<bool> > pooling4_conv5_connection;

	// - subnets -
	vector<conv_subnet3D*> input_conv1;
	vector<pooling_subnet2D*> conv1_pooling2;
	vector<conv_subnet3D*> pooling2_conv3;
	vector<pooling_subnet2D*> conv3_pooling4;
	vector<conv_subnet3D*> pooling4_conv5;
	mcp_bprop* conv5_hidden6;
	mcp_bprop* hidden6_output;	
	
	// - parameters -
	error_metric metric;
	float error_threshold;
	float weight_rnd_min;
	float weight_rnd_max;
	int epochs_for_hessian_estimation;
	int nb_sampled_patterns;

	lenet5(parameters*);
	~lenet5();
	// the input feature maps may be more than one;
	void load();
	void dump();
	void forward(bool);
	float train(int, data_set_mnist*, bool);
	float train_back_propagation(data_set_mnist*, bool);
	float test_mnist(data_set_mnist*);
	void clear_hessian_information();
	void hessian_estimation(data_set_mnist*);
	int judgement();


	//Added by Liqiang He at Feb. 21, 2017

	//OpenCL specific variables and function
	void initializeOpenCL();

	std::vector<cl::Platform>	platforms;
	std::vector<cl::Device>		devices;
	cl::Context			context;
	cl::CommandQueue		queue;
	cl::Program			program;
	cl_int				opencl_err;
	cl::Event 			event;

	std::string			kernel_source_path;
	std::string			kernel_binary_path;
	
	//Method running on GPU
	int use_gpu;
	bool in_has_bias;

	//input layer, conv1 layer, and synapses between two layers
	float *h_all_input_neurons;	// to store all the input neurons in the data set
	float *h_input_conv1_synapses_values;
	float *h_input_conv1_synapses_hessian;

	cl::Buffer d_test;

	cl::Buffer d_all_input_neurons;
	cl::Buffer d_conv1_neurons;
	cl::Buffer d_input_conv1_synapses_values;
	cl::Buffer d_input_conv1_synapses_hessian;
	cl::Buffer d_input_conv1_derivatives_out;
	cl::Buffer d_input_conv1_gradients_kernel;
	cl::Buffer d_input_conv1_gradients_bias;
	cl::Buffer d_input_conv1_gradients_out;
	cl::Buffer d_input_conv1_second_gradients_out_sum;
	cl::Buffer d_input_conv1_second_gradients_out;
	cl::Buffer d_input_conv1_fin_temp;

	// conv1-pooling2 layer
	float *h_conv1_pooling2_bias_weight;
	float *h_conv1_pooling2_bias_weight_hessian;
	float *h_conv1_pooling2_coefficient;
	float *h_conv1_pooling2_coefficient_hessian;

	cl::Buffer d_pooling2_neurons;
	cl::Buffer d_conv1_pooling2_bias_weight;
	cl::Buffer d_conv1_pooling2_bias_weight_hessian;
	cl::Buffer d_conv1_pooling2_coefficient;
	cl::Buffer d_conv1_pooling2_coefficient_hessian;
	cl::Buffer d_conv1_pooling2_derivatives_out;
	cl::Buffer d_conv1_pooling2_gradients_out;
	cl::Buffer d_conv1_pooling2_second_gradients_out;
	cl::Buffer d_conv1_pooling2_input_sampledown;

	//pooling2, conv3,
	float *h_pooling2_conv3_synapses_values;
	float *h_pooling2_conv3_synapses_hessian;

	cl::Buffer d_conv3_neurons;
	cl::Buffer d_pooling2_conv3_synapses_values;
	cl::Buffer d_pooling2_conv3_synapses_hessian;
	cl::Buffer d_pooling2_conv3_derivatives_out;
	cl::Buffer d_pooling2_conv3_gradients_kernel;
	cl::Buffer d_pooling2_conv3_gradients_bias;
	cl::Buffer d_pooling2_conv3_gradients_out;
	cl::Buffer d_pooling2_conv3_second_gradients_out_sum;
	cl::Buffer d_pooling2_conv3_second_gradients_out;
	cl::Buffer d_pooling2_conv3_fin_temp;

	// conv3-pooling4 layer
	float *h_conv3_pooling4_bias_weight;
	float *h_conv3_pooling4_bias_weight_hessian;
	float *h_conv3_pooling4_coefficient;
	float *h_conv3_pooling4_coefficient_hessian;

	cl::Buffer d_pooling4_neurons;
	cl::Buffer d_conv3_pooling4_bias_weight;
	cl::Buffer d_conv3_pooling4_bias_weight_hessian;
	cl::Buffer d_conv3_pooling4_coefficient;
	cl::Buffer d_conv3_pooling4_coefficient_hessian;
	cl::Buffer d_conv3_pooling4_derivatives_out;
	cl::Buffer d_conv3_pooling4_gradients_out;
	cl::Buffer d_conv3_pooling4_second_gradients_out;
	cl::Buffer d_conv3_pooling4_input_sampledown;

	//pooling4, conv5,
	float *h_pooling4_conv5_synapses_values;
	float *h_pooling4_conv5_synapses_hessian;

	cl::Buffer d_conv5_neurons;
	cl::Buffer d_pooling4_conv5_synapses_values;
	cl::Buffer d_pooling4_conv5_synapses_hessian;
	cl::Buffer d_pooling4_conv5_derivatives_out;
	cl::Buffer d_pooling4_conv5_gradients_kernel;
	cl::Buffer d_pooling4_conv5_gradients_bias;
	cl::Buffer d_pooling4_conv5_gradients_out;
	cl::Buffer d_pooling4_conv5_second_gradients_out_sum;
	cl::Buffer d_pooling4_conv5_second_gradients_out;
	cl::Buffer d_pooling4_conv5_fin_temp;

	// conv5-hidden6 layer
	float *h_hidden6_neurons;
	float *h_conv5_hidden6_synapses_values;
	float *h_conv5_hidden6_synapses_hessian;

	cl::Buffer d_hidden6_neurons;
	cl::Buffer d_conv5_hidden6_synapses_values;
	cl::Buffer d_conv5_hidden6_synapses_hessian;
	cl::Buffer d_conv5_hidden6_derivatives_out;
	cl::Buffer d_conv5_hidden6_gradients_out;
	cl::Buffer d_conv5_hidden6_second_gradients_out;

	float *h_all_output_neurons;
	float *h_hidden6_output_synapses_values;
	float *h_hidden6_output_synapses_hessian;

	cl::Buffer d_all_output_neurons;
	cl::Buffer d_hidden6_output_synapses_values;
	cl::Buffer d_hidden6_output_synapses_hessian;
	cl::Buffer d_hidden6_output_derivatives_out;
	cl::Buffer d_hidden6_output_gradients_out;
	cl::Buffer d_hidden6_output_second_gradients_out;

	// back propagation
	float *h_all_row_outputs;
	cl::Buffer d_all_row_outputs;

	void forward_gpu(bool, int);
	float train_gpu(int, data_set_mnist*, bool);
	float train_back_propagation_gpu(data_set_mnist*, bool);
	float test_mnist_gpu(data_set_mnist*);
	void clear_hessian_information_gpu();
	void hessian_estimation_gpu(data_set_mnist*);
	int judgement_gpu(data_set_mnist* dataset);
	//End of Liqiang He
};

#endif
