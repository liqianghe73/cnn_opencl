/*
  2013-07-15 (Yuan Gao)
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <string>

//#include <time.h>

#include "../lib/net.hh"
#include "../lib/neurons.hh"
#include "../lib/synapses.hh"
#include "../lib/parameters.hh"
#include "../lib/data.hh"
#include "../lib/tools.hh"
#include "../input/data_mnist.hh"
#include "../subnets/mcp_bprop.hh"

#include "lenet5.hh"

using namespace std;

extern bool dbg;

lenet5::lenet5(parameters* _params)
  : net(_params)
{
  cout << "--- in lenet5 constructor ---" << endl;
  bool dbg_load_synapses = false;

  // - defaults -
  verbose = false;

  // - parameters -
  params->set_float("weight_rnd_min", -0.1);
  params->set_float("weight_rnd_max", 0.1);
  metric = (error_metric)(_params->get_int("metric"));
  epochs_for_hessian_estimation = params->get_int("epochs for hessian estimation");
  nb_sampled_patterns = params->get_int("number of patterns for hessian estimation");
  //srand((unsigned)time(0));

  use_gpu = params->get_int("use_gpu");

  string kernel_file_path = params->get_string("kernel_path");
  
  if(kernel_file_path.find(".cl") != std::string::npos)
	kernel_source_path = kernel_file_path;
  else
	kernel_binary_path = kernel_file_path;

  if(use_gpu)
	initializeOpenCL();

  in_has_bias = params->get_bool("in_has_bias");

  // - create network -
 
  // - input layer -
  cout << "--- 1.1 creating input neurons ---" << endl;  
  // There is only one feature map in input layer;
  input_neurons.resize(params->get_int("nb_featuremap_input"));
  // add set of neurons2D to layer (int _size_x, int _size_y, is_input=true, is_output=false);
  input_neurons.at(0) = new neurons2D(params->get_int("size_x_input"), params->get_int("size_y_input"), true, false);
 
  // - convolutional layer C1 -
  cout << "--- 1.2 creating C1 neurons ---" << endl;  
  // There are 6 feature maps in C1;
  conv1_neurons.resize(params->get_int("nb_featuremap_conv1"));
  // (int _size_x, int _size_y, is_input=false, is_output=false);
  for (int i = 0; i < conv1_neurons.size(); i++)
 	 conv1_neurons.at(i) = new neurons2D(params->get_int("size_x_conv1"), params->get_int("size_y_conv1"), false, false);

  // synapses (between input and C1 layers);
  cout << "--- 1.3 creating input-C1 synapses ---" << endl;  
  // synapses (int Kx, int Ky)
  input_conv1_synapses.resize(params->get_int("nb_featuremap_conv1"));
  for (int fo = 0; fo < conv1_neurons.size(); fo++)
  {
    input_conv1_synapses.at(fo).resize(params->get_int("nb_featuremap_input"));
    for (int fi = 0; fi < input_neurons.size(); fi++)
      input_conv1_synapses.at(fo).at(fi) = new synapses(params->get_int("size_x_conv_kernel"), params->get_int("size_y_conv_kernel"));
  }


  // create subnets between input and C1 layers;
  cout << "--- 1.4 creating input-C1 subnets ---" << endl; 
  // indicate that there is an input bias neuron, the bias is in the subnet;
  // conv_subnet3D (_name, _params, _fout, _fins, _syns, _has_bias)
  input_conv1.resize(params->get_int("nb_featuremap_conv1"));
  for (int fo = 0; fo < conv1_neurons.size(); fo++)
    //input_conv1.at(fo) = new conv_subnet3D("input_conv1_" + tools::int2string(fo), params, conv1_neurons.at(fo), input_neurons, input_conv1_synapses.at(fo), true);
    input_conv1.at(fo) = new conv_subnet3D("input_conv1_" + tools::int2string(fo), params, conv1_neurons.at(fo), input_neurons, input_conv1_synapses.at(fo), in_has_bias);


  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_conv1_neurons = params->get_int("nb_featuremap_conv1") * params->get_int("size_y_conv1") * params->get_int("size_x_conv1");

	int size_of_h_input_conv1_synapses;
	if(in_has_bias) 
		size_of_h_input_conv1_synapses = (params->get_int("nb_featuremap_conv1") * params->get_int("nb_featuremap_input") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv1")); // the last part is for biases of each featuremap in conv1
	else 
		size_of_h_input_conv1_synapses = (params->get_int("nb_featuremap_conv1") * params->get_int("nb_featuremap_input") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel"));
	//initialize the synapses
	h_input_conv1_synapses_values = (float*)malloc(sizeof(float) * size_of_h_input_conv1_synapses); //has bias
	h_input_conv1_synapses_hessian = (float*)malloc(sizeof(float) * size_of_h_input_conv1_synapses); //has bias
  	for (int fo = 0; fo < conv1_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < input_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++)
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++)
					h_input_conv1_synapses_values[fo * input_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  input_conv1_synapses.at(fo).at(fi)->values.at(fj).at(fk);
    		}
  	}
	if(in_has_bias) {
  		for (int fo = 0; fo < conv1_neurons.size(); fo++)
			// the organization is : [ nb_featuremap_conv1 * nb_featuremap_input * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv1
			h_input_conv1_synapses_values[(size_of_h_input_conv1_synapses - conv1_neurons.size()) + fo] = input_conv1.at(fo)->bias_weight; 
	}

	//allocate device memory
	//allocate device space for synapses and init the values
	d_conv1_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_conv1_neurons * sizeof(float), NULL, &opencl_err);
	d_input_conv1_synapses_values = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_input_conv1_synapses * sizeof(float), NULL, &opencl_err);
	d_input_conv1_synapses_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_input_conv1_synapses * sizeof(float), NULL, &opencl_err);
	d_input_conv1_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv1") * params->get_int("size_y_conv1") * params->get_int("size_x_conv1") * sizeof(float), NULL, &opencl_err);
	d_input_conv1_gradients_kernel = cl::Buffer(context, CL_MEM_READ_WRITE, (params->get_int("nb_featuremap_conv1") * params->get_int("nb_featuremap_input") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel")) * sizeof(float), NULL, &opencl_err);
	d_input_conv1_gradients_bias = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv1") * sizeof(float), NULL, &opencl_err);
	d_input_conv1_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv1") * params->get_int("size_y_conv1") * params->get_int("size_x_conv1") * sizeof(float), NULL, &opencl_err);
	d_input_conv1_second_gradients_out_sum = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv1") * sizeof(float), NULL, &opencl_err);
	d_input_conv1_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv1") * params->get_int("size_y_conv1") * params->get_int("size_x_conv1") * sizeof(float), NULL, &opencl_err);

	opencl_err = queue.enqueueWriteBuffer(d_input_conv1_synapses_values, CL_TRUE, 0, size_of_h_input_conv1_synapses * sizeof(float), h_input_conv1_synapses_values, NULL, &event);
	queue.finish();
  }  

  // - pooling layer P2 -
  cout << "--- 2.1 creating P2 neurons ---" << endl;  
  // There are 6 feature maps in P2;
  pooling2_neurons.resize(params->get_int("nb_featuremap_pooling2"));
  // (int _size_x, int _size_y, is_input=false, is_output=false);
  for (int i = 0; i < pooling2_neurons.size(); i++)
    pooling2_neurons.at(i) = new neurons2D(params->get_int("size_x_pooling2"), params->get_int("size_y_pooling2"), false, false);
  
  // create subnets between C1 and P2 layers;
  cout << "--- 2.2 creating C1-P2 subnets ---" << endl;  
  // indicate that there is an input bias neuron, the bias is in the subnet;
  // pooling_subnet2D (_name, _params, _conv_subin, _fout, _syn, _has_bias)
  conv1_pooling2.resize(params->get_int("nb_featuremap_pooling2"));
  for (int fo = 0; fo < pooling2_neurons.size(); fo++)
    //conv1_pooling2.at(fo) = new pooling_subnet2D("conv1_pooling2_" + tools::int2string(fo), "M", params, input_conv1.at(fo), pooling2_neurons.at(fo), true);
    conv1_pooling2.at(fo) = new pooling_subnet2D("conv1_pooling2_" + tools::int2string(fo), "M", params, input_conv1.at(fo), pooling2_neurons.at(fo), in_has_bias);
  
  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_pooling2_neurons = params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") ;

	if(in_has_bias) {
		h_conv1_pooling2_bias_weight = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling2"));
		h_conv1_pooling2_bias_weight_hessian = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling2"));
  		for (int fo = 0; fo < pooling2_neurons.size(); fo++)
			h_conv1_pooling2_bias_weight[fo] = conv1_pooling2.at(fo)->bias_weight; 
	}
		
	h_conv1_pooling2_coefficient = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling2"));
	h_conv1_pooling2_coefficient_hessian = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling2"));
  	for (int fo = 0; fo < pooling2_neurons.size(); fo++)
		h_conv1_pooling2_coefficient[fo] = conv1_pooling2.at(fo)->coefficient; 

	//allocate device memory
	d_pooling2_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling2_neurons * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_bias_weight = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_bias_weight_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_coefficient = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_coefficient_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") * sizeof(float), NULL, &opencl_err);
	d_conv1_pooling2_input_sampledown = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") * sizeof(float), NULL, &opencl_err);

	// copy data to gpu
	opencl_err = queue.enqueueWriteBuffer(d_conv1_pooling2_bias_weight, CL_TRUE, 0, params->get_int("nb_featuremap_pooling2") * sizeof(float), h_conv1_pooling2_bias_weight, NULL, &event);
	opencl_err = queue.enqueueWriteBuffer(d_conv1_pooling2_coefficient, CL_TRUE, 0, params->get_int("nb_featuremap_pooling2") * sizeof(float), h_conv1_pooling2_coefficient, NULL, &event);
	queue.finish();
  }

  // - convolutional layer C3 -
  cout << "--- 3.1 creating C3 neurons ---" << endl;  
  // There are 16 feature maps in C3;
  conv3_neurons.resize(params->get_int("nb_featuremap_conv3"));
  // (int _size_x, int _size_y, is_input=false, is_output=false);
  for (int i = 0; i < conv3_neurons.size(); i++)
    conv3_neurons.at(i) = new neurons2D(params->get_int("size_x_conv3"), params->get_int("size_y_conv3"), false, false);

  // set the connection table between P2 and C3 layers;
  pooling2_conv3_connection.resize(params->get_int("nb_featuremap_conv3"));
  for (int fo = 0; fo < conv3_neurons.size(); fo++)
  {
    pooling2_conv3_connection.at(fo).resize(params->get_int("nb_featuremap_pooling2"));
    for (int fi = 0; fi < pooling2_neurons.size(); fi++)
      pooling2_conv3_connection.at(fo).at(fi) = true;
  }

  // synapses (between P2 and C3 layers);
  cout << "--- 3.2 creating P2-C3 synapses ---" << endl;
  // synapses (int Kx, int Ky)
  pooling2_conv3_synapses.resize(params->get_int("nb_featuremap_conv3"));
  for (int fo = 0; fo < conv3_neurons.size(); fo++)
  {
    pooling2_conv3_synapses.at(fo).resize(params->get_int("nb_featuremap_pooling2"));
    for (int fi = 0; fi < pooling2_neurons.size(); fi++)
    {
      if( pooling2_conv3_connection.at(fo).at(fi) == true)
        pooling2_conv3_synapses.at(fo).at(fi) = new synapses(params->get_int("size_x_conv_kernel"), params->get_int("size_y_conv_kernel"));
      else
        pooling2_conv3_synapses.at(fo).at(fi) = NULL;
    }
  }

  // create subnets between P2 and C3 layers;
  cout << "--- 3.3 creating P2-C3 subnets ---" << endl;
  // indicate that there is an input bias neuron, the bias is in the subnet;
  // conv_subnet3D (_name, _params, _fout, _pooling_subins, _syns, _has_bias)
  pooling2_conv3.resize(params->get_int("nb_featuremap_conv3"));
  for (int fo = 0; fo < conv3_neurons.size(); fo++)
    //pooling2_conv3.at(fo) = new conv_subnet3D("pooling2_conv3_" + tools::int2string(fo), params, conv3_neurons.at(fo), conv1_pooling2, pooling2_conv3_synapses.at(fo), true);
    pooling2_conv3.at(fo) = new conv_subnet3D("pooling2_conv3_" + tools::int2string(fo), params, conv3_neurons.at(fo), conv1_pooling2, pooling2_conv3_synapses.at(fo), in_has_bias);
  
  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_conv3_neurons = params->get_int("nb_featuremap_conv3") * params->get_int("size_y_conv3") * params->get_int("size_x_conv3");

	int size_of_h_pooling2_conv3_synapses;
	if(in_has_bias) 
		size_of_h_pooling2_conv3_synapses = (params->get_int("nb_featuremap_conv3") * params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv3")); // the last part is for biases of each featuremap in conv3
	else 
		size_of_h_pooling2_conv3_synapses = (params->get_int("nb_featuremap_conv3") * params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel"));
	//initialize the synapses
	h_pooling2_conv3_synapses_values = (float*)malloc(sizeof(float) * size_of_h_pooling2_conv3_synapses); //has bias
	h_pooling2_conv3_synapses_hessian = (float*)malloc(sizeof(float) * size_of_h_pooling2_conv3_synapses); //has bias
  	for (int fo = 0; fo < conv3_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < pooling2_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++)
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++) {
      				  if( pooling2_conv3_connection.at(fo).at(fi) == true)
					h_pooling2_conv3_synapses_values[fo * pooling2_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  pooling2_conv3_synapses.at(fo).at(fi)->values.at(fj).at(fk);
				  else
					h_pooling2_conv3_synapses_values[fo * pooling2_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] = 0;
				}
    		}
  	}
	if(in_has_bias) {
  		for (int fo = 0; fo < conv3_neurons.size(); fo++) {
			// the organization is : [ nb_featuremap_conv3 * nb_featuremap_pooling2 * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv3] 
			h_pooling2_conv3_synapses_values[(size_of_h_pooling2_conv3_synapses - conv3_neurons.size()) + fo] = pooling2_conv3.at(fo)->bias_weight; 
		}
	}

	//allocate device memory
	//allocate device space for synapses and init the values
	d_conv3_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_conv3_neurons * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_synapses_values = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling2_conv3_synapses * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_synapses_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling2_conv3_synapses * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * params->get_int("size_y_conv3") * params->get_int("size_x_conv3") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_gradients_kernel = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_gradients_bias = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * params->get_int("size_y_conv3") * params->get_int("size_x_conv3") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_second_gradients_out_sum = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * params->get_int("size_y_conv3") * params->get_int("size_x_conv3") * sizeof(float), NULL, &opencl_err);
	d_pooling2_conv3_fin_temp = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv3") * params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_pooling2") * params->get_int("size_x_pooling2") * sizeof(float), NULL, &opencl_err);

	opencl_err = queue.enqueueWriteBuffer(d_pooling2_conv3_synapses_values, CL_TRUE, 0, size_of_h_pooling2_conv3_synapses * sizeof(float), h_pooling2_conv3_synapses_values, NULL, &event);
	queue.finish();
  }  

  // - pooling layer P4 -
  cout << "--- 4.1 creating P4 neurons ---" << endl; 
  // There are 16 feature maps in P4;
  pooling4_neurons.resize(params->get_int("nb_featuremap_pooling4"));
  // (int _size_x, int _size_y, is_input=false, is_output=false);
  for (int i = 0; i < pooling4_neurons.size(); i++)
 	pooling4_neurons.at(i) = new neurons2D(params->get_int("size_x_pooling4"), params->get_int("size_y_pooling4"), false, false);

  // create subnet between C3 and P4 layers;
  cout << "--- 4.2 creating C3-P4 subnets ---" << endl;
  // indicate that there is an input bias neuron, the bias is in the subnet;
  // // pooling_subnet2D (_name, _params, _conv_subin, _fout, _syn, _has_bias)
  conv3_pooling4.resize(params->get_int("nb_featuremap_pooling4"));
  for (int fo = 0; fo < pooling4_neurons.size(); fo++)
    //conv3_pooling4.at(fo) = new pooling_subnet2D("conv3_pooling4_" + tools::int2string(fo), "M", params, pooling2_conv3.at(fo), pooling4_neurons.at(fo), true);
    conv3_pooling4.at(fo) = new pooling_subnet2D("conv3_pooling4_" + tools::int2string(fo), "M", params, pooling2_conv3.at(fo), pooling4_neurons.at(fo), in_has_bias);

  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_pooling4_neurons = params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4");

	if(in_has_bias) {
		h_conv3_pooling4_bias_weight = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling4"));
		h_conv3_pooling4_bias_weight_hessian = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling4"));
  		for (int fo = 0; fo < pooling4_neurons.size(); fo++)
			h_conv3_pooling4_bias_weight[fo] = conv3_pooling4.at(fo)->bias_weight; 
	}
		
	h_conv3_pooling4_coefficient = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling4"));
	h_conv3_pooling4_coefficient_hessian = (float*)malloc(sizeof(float) * params->get_int("nb_featuremap_pooling4"));
  	for (int fo = 0; fo < pooling4_neurons.size(); fo++)
		h_conv3_pooling4_coefficient[fo] = conv3_pooling4.at(fo)->coefficient; 

	//allocate device memory
	d_pooling4_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling4_neurons * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_bias_weight = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_bias_weight_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_coefficient = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_coefficient_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4") * sizeof(float), NULL, &opencl_err);
	d_conv3_pooling4_input_sampledown = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4") * sizeof(float), NULL, &opencl_err);

	// copy data to gpu
	opencl_err = queue.enqueueWriteBuffer(d_conv3_pooling4_bias_weight, CL_TRUE, 0, params->get_int("nb_featuremap_pooling4") * sizeof(float), h_conv3_pooling4_bias_weight, NULL, &event);
	opencl_err = queue.enqueueWriteBuffer(d_conv3_pooling4_coefficient, CL_TRUE, 0, params->get_int("nb_featuremap_pooling4") * sizeof(float), h_conv3_pooling4_coefficient, NULL, &event);
	queue.finish();
  }

  // - convolutional layer C5 -
  cout << "--- 5.1 creating C5 neurons ---" << endl;
  // There are 120 feature maps in C5;
  conv5_neurons.resize(params->get_int("nb_featuremap_conv5"));
  // (int _size_x, int _size_y, is_input=false, is_output=false);
  for (int i = 0; i < conv5_neurons.size(); i++)
    conv5_neurons.at(i) = new neurons2D(params->get_int("size_x_conv5"), params->get_int("size_y_conv5"), false, false);

  // set the connection table between P4 and C5 layers;
  pooling4_conv5_connection.resize(params->get_int("nb_featuremap_conv5"));
  for (int fo = 0; fo < conv5_neurons.size(); fo++)
  {
    pooling4_conv5_connection.at(fo).resize(params->get_int("nb_featuremap_pooling4"));
    for (int fi = 0; fi < pooling4_neurons.size(); fi++)
      pooling4_conv5_connection.at(fo).at(fi) = true;
  }

  // synapses (between P4 and C5 layers);
  cout << "--- 5.2 creating P4-C5 synapses ---" << endl;
  // synapses (int Kx, int Ky)
  pooling4_conv5_synapses.resize(params->get_int("nb_featuremap_conv5"));
  for (int fo = 0; fo < conv5_neurons.size(); fo++)
  {
    pooling4_conv5_synapses.at(fo).resize(params->get_int("nb_featuremap_pooling4"));
    for (int fi = 0; fi < pooling4_neurons.size(); fi++)
    {
      if( pooling4_conv5_connection.at(fo).at(fi) == true)
        pooling4_conv5_synapses.at(fo).at(fi) = new synapses(params->get_int("size_x_conv_kernel"), params->get_int("size_y_conv_kernel"));
      else
        pooling4_conv5_synapses.at(fo).at(fi) = NULL;
    }
  }

  // create subnets between P4 and C5 layers;
  cout << "--- 5.3 creating P4-C5 subnets ---" << endl;
  // indicate that there is an input bias neuron, the bias is in the subnet;
  // conv_subnet3D (_name, _params, _fout, _pooling_subins, _syns, _has_bias)
  pooling4_conv5.resize(params->get_int("nb_featuremap_conv5"));
  for (int fo = 0; fo < conv5_neurons.size(); fo++)
    //pooling4_conv5.at(fo) = new conv_subnet3D("pooling4_conv5_" + tools::int2string(fo), params, conv5_neurons.at(fo), conv3_pooling4, pooling4_conv5_synapses.at(fo), true);
    pooling4_conv5.at(fo) = new conv_subnet3D("pooling4_conv5_" + tools::int2string(fo), params, conv5_neurons.at(fo), conv3_pooling4, pooling4_conv5_synapses.at(fo), in_has_bias);
  
  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_conv5_neurons = params->get_int("nb_featuremap_conv5") * params->get_int("size_y_conv5") * params->get_int("size_x_conv5");

	int size_of_h_pooling4_conv5_synapses;
	if(in_has_bias) 
		size_of_h_pooling4_conv5_synapses = (params->get_int("nb_featuremap_conv5") * params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv5")); // the last part is for biases or hessian of each featuremap in conv5
	else 
		size_of_h_pooling4_conv5_synapses = (params->get_int("nb_featuremap_conv5") * params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel"));
	//initialize the synapses
	h_pooling4_conv5_synapses_values = (float*)malloc(sizeof(float) * size_of_h_pooling4_conv5_synapses); //has bias
	h_pooling4_conv5_synapses_hessian = (float*)malloc(sizeof(float) * size_of_h_pooling4_conv5_synapses); //has bias
  	for (int fo = 0; fo < conv5_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < pooling4_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++)
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++) {
      				  if( pooling4_conv5_connection.at(fo).at(fi) == true)
					h_pooling4_conv5_synapses_values[fo * pooling4_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  pooling4_conv5_synapses.at(fo).at(fi)->values.at(fj).at(fk);
				  else
					h_pooling4_conv5_synapses_values[fo * pooling4_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] = 0;
				}
    		}
  	}
	if(in_has_bias) {
  		for (int fo = 0; fo < conv5_neurons.size(); fo++) {
			// the organization is : [ nb_featuremap_conv5 * nb_featuremap_pooling4 * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv5] 
			h_pooling4_conv5_synapses_values[(size_of_h_pooling4_conv5_synapses - conv5_neurons.size()) + fo] = pooling4_conv5.at(fo)->bias_weight; 
		}
	}

	//allocate device memory
	//allocate device space for synapses and init the values
	d_conv5_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_conv5_neurons * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_synapses_values = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling4_conv5_synapses * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_synapses_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_pooling4_conv5_synapses * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * params->get_int("size_y_conv5") * params->get_int("size_x_conv5") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_gradients_kernel = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_gradients_bias = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * params->get_int("size_y_conv5") * params->get_int("size_x_conv5") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_second_gradients_out_sum = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * params->get_int("size_y_conv5") * params->get_int("size_x_conv5") * sizeof(float), NULL, &opencl_err);
	d_pooling4_conv5_fin_temp = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_featuremap_conv5") * params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_pooling4") * params->get_int("size_x_pooling4") * sizeof(float), NULL, &opencl_err);


	opencl_err = queue.enqueueWriteBuffer(d_pooling4_conv5_synapses_values, CL_TRUE, 0, size_of_h_pooling4_conv5_synapses * sizeof(float), h_pooling4_conv5_synapses_values, NULL, &event);
	queue.finish();
  }  

  // - hidden layer H6 -
  cout << "--- 6.1 creating H6 neurons ---" << endl;
  // There are no feature maps in H6, but 84 neurons;
  // (int nb_neurons, has_bias = true, is_input=false, is_output=false);
  hidden6_neurons = new neurons(params->get_int("nb_neuron_hidden6"), true, false, false);
  // synapses (between C5 and H6 layers);
  cout << "--- 6.2 creating C5-H6 synapses ---" << endl;
  // synapses (int size_in, int size_out, bool _in_has_bias)
  conv5_hidden6_synapse = new synapses(params->get_int("nb_featuremap_conv5"), params->get_int("nb_neuron_hidden6"), true);
  
  // create subnet between C5 and H6 layers;
  cout << "--- 6.3 creating C5-H6 subnets ---" << endl;
  // mcp_bprop (_name, _params, _conv_subins, _	has_neurons2D_bias, _nout, _syn)
  conv5_hidden6 = new mcp_bprop("conv5_hidden6", params, pooling4_conv5, true, hidden6_neurons, conv5_hidden6_synapse);
  
  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_hidden6_neurons;
	if(in_has_bias)
		size_of_h_hidden6_neurons = params->get_int("nb_neuron_hidden6") + 1;
	else
		size_of_h_hidden6_neurons = params->get_int("nb_neuron_hidden6");
	h_hidden6_neurons = (float*)malloc(sizeof(float) * size_of_h_hidden6_neurons);
	for (int fo = 0; fo < size_of_h_hidden6_neurons; fo++)
		h_hidden6_neurons[fo] = 0;
	if(in_has_bias) h_hidden6_neurons[size_of_h_hidden6_neurons - 1] = 1;


	int size_of_h_conv5_hidden6_synapses;
	if(in_has_bias) 
		size_of_h_conv5_hidden6_synapses = (params->get_int("nb_neuron_hidden6") * params->get_int("nb_featuremap_conv5") + params->get_int("nb_neuron_hidden6")); // the last part is for biases of each neuron in hidden6
	else 
		size_of_h_conv5_hidden6_synapses = (params->get_int("nb_neuron_hidden6") * params->get_int("nb_featuremap_conv5"));
	//initialize the synapses
	h_conv5_hidden6_synapses_values = (float*)malloc(sizeof(float) * size_of_h_conv5_hidden6_synapses); //has bias
	h_conv5_hidden6_synapses_hessian = (float*)malloc(sizeof(float) * size_of_h_conv5_hidden6_synapses); //has bias
  	for (int fo = 0; fo < params->get_int("nb_neuron_hidden6"); fo++)
  	{
    		for (int fi = 0; fi < conv5_neurons.size(); fi++) {
			h_conv5_hidden6_synapses_values[fo * conv5_neurons.size() + fi] =  conv5_hidden6_synapse->values.at(fo).at(fi);
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < params->get_int("nb_neuron_hidden6"); fo++) {
			// the organization is : [ nb_neuron_hidden6 * nb_featuremap_conv5 * size_y_conv_kernel * size_x_conv_kernel + nb_neuron_hidden6] 
			h_conv5_hidden6_synapses_values[(size_of_h_conv5_hidden6_synapses - params->get_int("nb_neuron_hidden6")) + fo] = conv5_hidden6_synapse->values.at(fo).at(conv5_neurons.size());
		}
	}

	//allocate device memory
	d_hidden6_neurons = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_hidden6_neurons * sizeof(float), NULL, &opencl_err);
	d_conv5_hidden6_synapses_values = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_conv5_hidden6_synapses * sizeof(float), NULL, &opencl_err);
	d_conv5_hidden6_synapses_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_conv5_hidden6_synapses * sizeof(float), NULL, &opencl_err);
	d_conv5_hidden6_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_hidden6") * sizeof(float), NULL, &opencl_err);
	d_conv5_hidden6_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_hidden6") * sizeof(float), NULL, &opencl_err);
	d_conv5_hidden6_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_hidden6") * sizeof(float), NULL, &opencl_err);

	// copy data to gpu
	opencl_err = queue.enqueueWriteBuffer(d_hidden6_neurons, CL_TRUE, 0, size_of_h_hidden6_neurons * sizeof(float), h_hidden6_neurons, NULL, &event);
	opencl_err = queue.enqueueWriteBuffer(d_conv5_hidden6_synapses_values, CL_TRUE, 0, size_of_h_conv5_hidden6_synapses * sizeof(float), h_conv5_hidden6_synapses_values, NULL, &event);
	queue.finish();
  }

  // - output layer -
  cout << "--- 7.1 creating output neurons ---" << endl;
  // (has_bias=false, is_input=false, is_output=true);
  output_neurons = new neurons(params->get_int("nb_neuron_output"), false, false, true);
  cout << "--- 7.2 creating output synapses ---" << endl;
  hidden6_output_synapse = new synapses(params->get_int("nb_neuron_hidden6"), params->get_int("nb_neuron_output"), true);
  cout << "--- 7.3 creating output subnets ---" << endl;
  hidden6_output = new mcp_bprop("hidden6_output", params, conv5_hidden6, output_neurons, hidden6_output_synapse);

  if(use_gpu)
  {
	//allocate host memory
	int size_of_h_hidden6_output_synapses;
	if(in_has_bias) 
		size_of_h_hidden6_output_synapses = (params->get_int("nb_neuron_output") * params->get_int("nb_neuron_hidden6") + params->get_int("nb_neuron_output")); // the last part is for biases of each neuron in output
	else 
		size_of_h_hidden6_output_synapses = (params->get_int("nb_neuron_output") * params->get_int("nb_neuron_hidden6"));
	//initialize the synapses
	h_hidden6_output_synapses_values = (float*)malloc(sizeof(float) * size_of_h_hidden6_output_synapses); //has bias
	h_hidden6_output_synapses_hessian = (float*)malloc(sizeof(float) * size_of_h_hidden6_output_synapses); //has bias
  	for (int fo = 0; fo < params->get_int("nb_neuron_output"); fo++)
  	{
    		for (int fi = 0; fi < params->get_int("nb_neuron_hidden6"); fi++) {
			h_hidden6_output_synapses_values[fo * params->get_int("nb_neuron_hidden6") + fi] =  hidden6_output_synapse->values.at(fo).at(fi);
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < params->get_int("nb_neuron_output"); fo++) {
			// the organization is : [ nb_neuron_output * nb_neuron_hidden6 * size_y_conv_kernel * size_x_conv_kernel + nb_neuron_output] 
			h_hidden6_output_synapses_values[(size_of_h_hidden6_output_synapses - params->get_int("nb_neuron_output")) + fo] = hidden6_output_synapse->values.at(fo).at(params->get_int("nb_neuron_hidden6"));
		}
	}

	//allocate device memory
	d_hidden6_output_synapses_values = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_hidden6_output_synapses * sizeof(float), NULL, &opencl_err);
	d_hidden6_output_synapses_hessian = cl::Buffer(context, CL_MEM_READ_WRITE, size_of_h_hidden6_output_synapses * sizeof(float), NULL, &opencl_err);
	d_hidden6_output_derivatives_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_output") * sizeof(float), NULL, &opencl_err);
	d_hidden6_output_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_output") * sizeof(float), NULL, &opencl_err);
	d_hidden6_output_second_gradients_out = cl::Buffer(context, CL_MEM_READ_WRITE, params->get_int("nb_neuron_output") * sizeof(float), NULL, &opencl_err);

	// copy data to gpu
	opencl_err = queue.enqueueWriteBuffer(d_hidden6_output_synapses_values, CL_TRUE, 0, size_of_h_hidden6_output_synapses * sizeof(float), h_hidden6_output_synapses_values, NULL, &event);
	queue.finish();
  }
  cout << "--- created all instants of cnn ---" << endl;
  cout << "--- leave enet5 constructor ---" << endl;

}
	
lenet5::~lenet5() 
{  
  for (int i = 0; i < input_neurons.size(); i++)
    delete(input_neurons.at(i));
  for (int i = 0; i < conv1_neurons.size(); i++)
    delete(conv1_neurons.at(i));
  for (int i = 0; i < pooling2_neurons.size(); i++)
    delete(pooling2_neurons.at(i));
  for (int i = 0; i < conv3_neurons.size(); i++)
    delete(conv3_neurons.at(i));
  for (int i = 0; i < pooling4_neurons.size(); i++)
    delete(pooling4_neurons.at(i));
  for (int i = 0; i < conv5_neurons.size(); i++)
    delete(conv5_neurons.at(i));
  delete(hidden6_neurons);
  delete(output_neurons);

  for (int y = 0; y < input_conv1_synapses.size(); y++)
    for (int x = 0; x < input_conv1_synapses.at(y).size(); x++)
      delete(input_conv1_synapses.at(y).at(x));
  for (int y = 0; y < pooling2_conv3_synapses.size(); y++)
    for (int x = 0; x < pooling2_conv3_synapses.at(y).size(); x++)
      delete(pooling2_conv3_synapses.at(y).at(x));
  for (int y = 0; y < pooling4_conv5_synapses.size(); y++)
    for (int x = 0; x < pooling4_conv5_synapses.at(y).size(); x++)
      delete(pooling4_conv5_synapses.at(y).at(x));
  delete(conv5_hidden6_synapse);
  delete(hidden6_output_synapse);
  // add delete for those subnets???
  for (int i = 0; i < input_conv1.size(); i++)
    delete(input_conv1.at(i));
  for (int i = 0; i < conv1_pooling2.size(); i++)
    delete(conv1_pooling2.at(i));
  for (int i = 0; i < pooling2_conv3.size(); i++)
    delete(pooling2_conv3.at(i));
  for (int i = 0; i < conv3_pooling4.size(); i++)
    delete(conv3_pooling4.at(i));
  for (int i = 0; i < pooling4_conv5.size(); i++)
    delete(pooling4_conv5.at(i));
  delete(conv5_hidden6);
  delete(hidden6_output);

  if(use_gpu)
  {
	// input-conv1
	free(h_input_conv1_synapses_values);
	free(h_input_conv1_synapses_hessian);

	// conv1-pooling2
	free(h_conv1_pooling2_bias_weight);
	free(h_conv1_pooling2_bias_weight_hessian);
	free(h_conv1_pooling2_coefficient);
	free(h_conv1_pooling2_coefficient_hessian);

	// pooling2-conv3
	free(h_pooling2_conv3_synapses_values);
	free(h_pooling2_conv3_synapses_hessian);

	// conv3-pooling4
	free(h_conv3_pooling4_bias_weight);
	free(h_conv3_pooling4_bias_weight_hessian);
	free(h_conv3_pooling4_coefficient);
	free(h_conv3_pooling4_coefficient_hessian);

	// pooling4-conv5
	free(h_pooling4_conv5_synapses_values);
	free(h_pooling4_conv5_synapses_hessian);

	// conv5-hidden6
	free(h_hidden6_neurons);
	free(h_conv5_hidden6_synapses_values);
	free(h_conv5_hidden6_synapses_hessian);

	// hidden6-output
	free(h_hidden6_output_synapses_values);
	free(h_hidden6_output_synapses_hessian);

	// Release gpu memory objects
	if(d_conv1_neurons()) clReleaseMemObject(d_conv1_neurons());
#if 0
	clReleaseMemObject(d_input_conv1_synapses_values());
	clReleaseMemObject(d_input_conv1_synapses_hessian());
	clReleaseMemObject(d_input_conv1_derivatives_out());
	clReleaseMemObject(d_input_conv1_gradients_kernel());
	clReleaseMemObject(d_input_conv1_gradients_bias());
	clReleaseMemObject(d_input_conv1_gradients_out());
	clReleaseMemObject(d_input_conv1_second_gradients_out_sum());
	clReleaseMemObject(d_input_conv1_second_gradients_out());

	clReleaseMemObject(d_pooling2_neurons());
	clReleaseMemObject(d_conv1_pooling2_bias_weight());
	clReleaseMemObject(d_conv1_pooling2_bias_weight_hessian());
	clReleaseMemObject(d_conv1_pooling2_coefficient());
	clReleaseMemObject(d_conv1_pooling2_coefficient_hessian());
	clReleaseMemObject(d_conv1_pooling2_derivatives_out());
	clReleaseMemObject(d_conv1_pooling2_gradients_out());
	clReleaseMemObject(d_conv1_pooling2_second_gradients_out());
	clReleaseMemObject(d_conv1_pooling2_input_sampledown());
	
	clReleaseMemObject(d_conv3_neurons());
	clReleaseMemObject(d_pooling2_conv3_synapses_values());
	clReleaseMemObject(d_pooling2_conv3_synapses_hessian());
	clReleaseMemObject(d_pooling2_conv3_derivatives_out());
	clReleaseMemObject(d_pooling2_conv3_gradients_kernel());
	clReleaseMemObject(d_pooling2_conv3_gradients_bias());
	clReleaseMemObject(d_pooling2_conv3_gradients_out());
	clReleaseMemObject(d_pooling2_conv3_second_gradients_out_sum());
	clReleaseMemObject(d_pooling2_conv3_second_gradients_out());
	clReleaseMemObject(d_pooling2_conv3_fin_temp());

	clReleaseMemObject(d_pooling4_neurons());
	clReleaseMemObject(d_conv3_pooling4_bias_weight());
	clReleaseMemObject(d_conv3_pooling4_bias_weight_hessian());
	clReleaseMemObject(d_conv3_pooling4_coefficient());
	clReleaseMemObject(d_conv3_pooling4_coefficient_hessian());
	clReleaseMemObject(d_conv3_pooling4_derivatives_out());
	clReleaseMemObject(d_conv3_pooling4_gradients_out());
	clReleaseMemObject(d_conv3_pooling4_second_gradients_out());
	clReleaseMemObject(d_conv3_pooling4_input_sampledown());

	clReleaseMemObject(d_conv5_neurons());
	clReleaseMemObject(d_pooling4_conv5_synapses_values());
	clReleaseMemObject(d_pooling4_conv5_synapses_hessian());
	clReleaseMemObject(d_pooling4_conv5_derivatives_out());
	clReleaseMemObject(d_pooling4_conv5_gradients_kernel());
	clReleaseMemObject(d_pooling4_conv5_gradients_bias());
	clReleaseMemObject(d_pooling4_conv5_gradients_out());
	clReleaseMemObject(d_pooling4_conv5_second_gradients_out_sum());
	clReleaseMemObject(d_pooling4_conv5_second_gradients_out());
	clReleaseMemObject(d_pooling4_conv5_fin_temp());

	clReleaseMemObject(d_hidden6_neurons());
	clReleaseMemObject(d_conv5_hidden6_synapses_values());
	clReleaseMemObject(d_conv5_hidden6_synapses_hessian());
	clReleaseMemObject(d_conv5_hidden6_derivatives_out());
	clReleaseMemObject(d_conv5_hidden6_gradients_out());
	clReleaseMemObject(d_conv5_hidden6_second_gradients_out());

	clReleaseMemObject(d_hidden6_output_synapses_values());
	clReleaseMemObject(d_hidden6_output_synapses_hessian());
	clReleaseMemObject(d_hidden6_output_derivatives_out());
	clReleaseMemObject(d_hidden6_output_gradients_out());
	clReleaseMemObject(d_hidden6_output_second_gradients_out());

	// Release OpenCL variables
	clReleaseProgram(program());
	clReleaseCommandQueue(queue());
	clReleaseContext(context());
#endif
  }  
}

void lenet5::initializeOpenCL()
{
  cl::Platform::get(&platforms);
  cout << "number of platforms:" << platforms.size() << "\n";

  if (platforms.size() == 0) {
        cout << "Platform size 0\n";
	exit(1);
  }

  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

  context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cout << "number of devices " << devices.size() << "\n";

  queue = cl::CommandQueue(context, devices[0], 0);

  cout << "kernel_source:" << kernel_source_path << "\n";
  cout << "kernel_binary:" << kernel_binary_path << "\n";

  //Load program
  if(kernel_source_path == std::string("")) //use binary
  {
        std::ifstream kernel_binary(kernel_binary_path.c_str(), std::ios::binary);

        if ( kernel_binary.fail() )
        {
            cout << "Could not open kernel binary.\n";
        }

        kernel_binary.seekg(0, std::ios::end);

        int filesize = kernel_binary.tellg();

        kernel_binary.seekg(0, std::ios::beg);

        if ( filesize <= 0 )
        {
            cout << "The kernel binary has length 0?";
        }

        std::vector<unsigned char> binary_blob(filesize);
        kernel_binary.read(reinterpret_cast<char*>(&binary_blob[0]), filesize);
        kernel_binary.close();

        cl::Program::Binaries binaries(1, make_pair( (const void*) &binary_blob[0], (size_t) binary_blob.size()  ) );
        program = cl::Program( context, devices, binaries);
   }
  else // use source
  {
    	std::ifstream input_file(kernel_source_path.c_str());
    	std::istreambuf_iterator<char> end;
    	std::string src( std::istreambuf_iterator<char>(input_file), end );
    	input_file.close();

	cl::Program::Sources source(1, make_pair(src.c_str(),src.length()+1));
        program = cl::Program(context, source);
  }

  //Build program
  try
  {
	program.build(devices);
  	cout << " program.build with binary successful\n";
  }
  catch (cl::Error err) // print build log
  {
        cerr << " program.build with binary failed!\n";
        cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        throw err;
  }
}


void lenet5::dump()
{
  cout << "--- begin dump cnn ---" << endl;
  cout << "Begin dump input neurons:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_input"); fo++)
  {
    cout << "Input neurons: " << " Inputs to " << "Inputs.dump" << endl;
    ofstream file;
    file.open("./cnn_test_load_dump/Inputs.dump");
    file << input_neurons.at(0)->dump();
    file.close();   
  }

  cout << "Begin dump input_conv1:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
  {
    cout << "subnet: " << input_conv1.at(fo)->name << " to " << input_conv1.at(fo)->name + ".dump" << endl;
    ofstream file;
    file.open((dump_dir + input_conv1.at(fo)->name + ".dump").c_str());
    input_conv1.at(fo)->dump(file);
    file.close();   
  }
  
  cout << "Begin dump conv1_pooling2:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
  {
    cout << "subnet: " << conv1_pooling2.at(fo)->name << " to " << conv1_pooling2.at(fo)->name + ".dump" << endl;
    ofstream file;
    file.open((dump_dir + conv1_pooling2.at(fo)->name + ".dump").c_str());
    conv1_pooling2.at(fo)->dump(file);
    file.close();   
  }
  
  cout << "Begin dump pooling2_conv3:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
  {
    cout << "subnet: " << pooling2_conv3.at(fo)->name << " to " << pooling2_conv3.at(fo)->name + ".dump" << endl;
    ofstream file;
    file.open((dump_dir + pooling2_conv3.at(fo)->name + ".dump").c_str());
    pooling2_conv3.at(fo)->dump(file);
    file.close();   
  }

  cout << "Begin dump conv3_pooling4:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
  {
    cout << "subnet: " << conv3_pooling4.at(fo)->name << " to " << conv3_pooling4.at(fo)->name + ".dump" << endl;
    ofstream file;
    file.open((dump_dir + conv3_pooling4.at(fo)->name + ".dump").c_str());
    conv3_pooling4.at(fo)->dump(file);
    file.close();   
  }

  cout << "Begin dump pooling4_conv5:" << endl;
  for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
  {
    cout << "subnet: " << pooling4_conv5.at(fo)->name << " to " << pooling4_conv5.at(fo)->name + ".dump" << endl;
    ofstream file;
    file.open((dump_dir + pooling4_conv5.at(fo)->name + ".dump").c_str());
    pooling4_conv5.at(fo)->dump(file);
    file.close();   
  }

  cout << "Begin dump conv5_hidden6:" << endl;
  cout << "subnet: " << "conv5_hidden6" << " to " << "conv5_hidden6.dump" << endl;
  ofstream file;
  file.open("./cnn_test_load_dump/conv5_hidden6.dump");
  conv5_hidden6->dump(file);
  file.close();   

  cout << "Begin dump hidden6_output:" << endl;
  cout << "subnet: " << "hidden6_output" << " to " << "hidden6_output.dump" << endl;
  file.open("./cnn_test_load_dump/hidden6_output.dump");
  hidden6_output->dump(file);
  file.close();   
}

// load weights from dump files
void lenet5::load()
{
  // - input-C1 -
  cout << "--- 1 load input-C1 subnets ---" << endl;  
  // There are 6 subnets in input-C1;
  for (int fo = 0; fo < conv1_neurons.size(); fo++)
  {
    ifstream file;
    string sentence;
    file.open((dump_dir + input_conv1.at(fo)->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1 || sentence.find("synapses")!=-1)
      {
        float bias;
        sscanf((sentence).c_str(), "%*s %f", &bias);
	input_conv1.at(fo)->load(file, bias);
	break;
      }
    }
    file.close();   
  }

  // - pooling layer P2 -
  cout << "--- 2 load C1-P2 subnets ---" << endl;  
  // There are 6 subnets in C1-P2;
  for (int fo = 0; fo < pooling2_neurons.size(); fo++)
  {
    float bias;
    float coefficient;
    ifstream file;
    string sentence;
    file.open((dump_dir + conv1_pooling2.at(fo)->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1)
      {    
        sscanf((sentence).c_str(), "%*s %f", &bias);
      }
      if (sentence.find("coefficient")!=-1)
      {
        sscanf((sentence).c_str(), "%*s %f", &coefficient);
      }
    }
    conv1_pooling2.at(fo)->load(bias, coefficient);
    file.close();   
  }

  // - convolutional layer C3 -
  // There are 16 subnets in P2-C3;
  for (int fo = 0; fo < conv3_neurons.size(); fo++)
  {
    ifstream file;
    string sentence;
    file.open((dump_dir + pooling2_conv3.at(fo)->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1 || sentence.find("synapses")!=-1)
      {
        float bias;
        sscanf((sentence).c_str(), "%*s %f", &bias);
	pooling2_conv3.at(fo)->load(file, bias);
	break;
      }
    }
    file.close();   
  }

  // - pooling layer P4 -
  cout << "--- 3 load C3-P4 subnets ---" << endl;  
  // There are 16 subnets in C3-P4;
  for (int fo = 0; fo < pooling4_neurons.size(); fo++)
  {
    float bias;
    float coefficient;
    ifstream file;
    string sentence;
    file.open((dump_dir + conv3_pooling4.at(fo)->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1)
      {    
        sscanf((sentence).c_str(), "%*s %f", &bias);
      }
      if (sentence.find("coefficient")!=-1)
      {
        sscanf((sentence).c_str(), "%*s %f", &coefficient);
      }
    }
    conv3_pooling4.at(fo)->load(bias, coefficient);
    file.close();   
  }

  // - convolutional layer C5 -
  cout << "--- 4 load P4-C5 subnets ---" << endl;  
  // There are 8 subnets in P4-C5;
  for (int fo = 0; fo < conv5_neurons.size(); fo++)
  {
    ifstream file;
    string sentence;
    file.open((dump_dir + pooling4_conv5.at(fo)->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1 || sentence.find("synapses")!=-1)
      {
        float bias;
        sscanf((sentence).c_str(), "%*s %f", &bias);
	pooling4_conv5.at(fo)->load(file, bias);
	break;
      }
    }
    file.close();   
  }
  // - hidden layer H6 -
  cout << "--- 5 load C5-H6 subnets ---" << endl;  
    ifstream file;
    string sentence;
    file.open((dump_dir + conv5_hidden6->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1 || sentence.find("weight")!=-1)
      {
	conv5_hidden6->load(file);
	break;
      }
    }
    file.close();   

  // - output layer -
  cout << "--- 6 load H6-output subnets ---" << endl;  
    file.open((dump_dir + hidden6_output->name + ".dump").c_str());
    while (getline(file, sentence))
    {
      if (sentence.find("bias")!=-1 || sentence.find("weight")!=-1)
      {
	hidden6_output->load(file);
	break;
      }
    }
    file.close();   
  cout << "--- Load complete ---" << endl;

  if(use_gpu)
  {
   cout << "--- copy to gpu ---" << endl;

  	for (int fo = 0; fo < conv1_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < input_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++) {
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++) {
					h_input_conv1_synapses_values[fo * input_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  input_conv1_synapses.at(fo).at(fi)->values.at(fj).at(fk);
				}
			}
    		}
  	}
	int size_of_h_input_conv1_synapses;
	if(in_has_bias) {
		size_of_h_input_conv1_synapses = (params->get_int("nb_featuremap_conv1") * params->get_int("nb_featuremap_input") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv1")); // the last part is for biases of each featuremap in conv1
  		for (int fo = 0; fo < conv1_neurons.size(); fo++)
			// the organization is : [ nb_featuremap_conv1 * nb_featuremap_input * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv1
			h_input_conv1_synapses_values[(size_of_h_input_conv1_synapses - conv1_neurons.size()) + fo] = input_conv1.at(fo)->bias_weight; 
	}

	opencl_err = queue.enqueueWriteBuffer(d_input_conv1_synapses_values, CL_TRUE, 0, size_of_h_input_conv1_synapses * sizeof(float), h_input_conv1_synapses_values, NULL, &event);

	if(in_has_bias) {
  		for (int fo = 0; fo < pooling2_neurons.size(); fo++)
			h_conv1_pooling2_bias_weight[fo] = conv1_pooling2.at(fo)->bias_weight; 
	}
		
  	for (int fo = 0; fo < pooling2_neurons.size(); fo++)
		h_conv1_pooling2_coefficient[fo] = conv1_pooling2.at(fo)->coefficient; 

	opencl_err = queue.enqueueWriteBuffer(d_conv1_pooling2_bias_weight, CL_TRUE, 0, params->get_int("nb_featuremap_pooling2") * sizeof(float), h_conv1_pooling2_bias_weight, NULL, &event);
	opencl_err = queue.enqueueWriteBuffer(d_conv1_pooling2_coefficient, CL_TRUE, 0, params->get_int("nb_featuremap_pooling2") * sizeof(float), h_conv1_pooling2_coefficient, NULL, &event);

	int size_of_h_pooling2_conv3_synapses;
	if(in_has_bias) 
		size_of_h_pooling2_conv3_synapses = (params->get_int("nb_featuremap_conv3") * params->get_int("nb_featuremap_pooling2") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv3")); // the last part is for biases of each featuremap in conv3

  	for (int fo = 0; fo < conv3_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < pooling2_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++)
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++) {
      				  if( pooling2_conv3_connection.at(fo).at(fi) == true)
					h_pooling2_conv3_synapses_values[fo * pooling2_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  pooling2_conv3_synapses.at(fo).at(fi)->values.at(fj).at(fk);
				  else
					h_pooling2_conv3_synapses_values[fo * pooling2_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] = 0;
				}
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < conv3_neurons.size(); fo++) {
			// the organization is : [ nb_featuremap_conv3 * nb_featuremap_pooling2 * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv3] 
			h_pooling2_conv3_synapses_values[(size_of_h_pooling2_conv3_synapses - conv3_neurons.size()) + fo] = pooling2_conv3.at(fo)->bias_weight; 
		}
	}

	opencl_err = queue.enqueueWriteBuffer(d_pooling2_conv3_synapses_values, CL_TRUE, 0, size_of_h_pooling2_conv3_synapses * sizeof(float), h_pooling2_conv3_synapses_values, NULL, &event);

	if(in_has_bias) {
  		for (int fo = 0; fo < pooling4_neurons.size(); fo++)
			h_conv3_pooling4_bias_weight[fo] = conv3_pooling4.at(fo)->bias_weight; 
	}
		
  	for (int fo = 0; fo < pooling4_neurons.size(); fo++)
		h_conv3_pooling4_coefficient[fo] = conv3_pooling4.at(fo)->coefficient; 

	opencl_err = queue.enqueueWriteBuffer(d_conv3_pooling4_bias_weight, CL_TRUE, 0, params->get_int("nb_featuremap_pooling4") * sizeof(float), h_conv3_pooling4_bias_weight, NULL, &event);
	opencl_err = queue.enqueueWriteBuffer(d_conv3_pooling4_coefficient, CL_TRUE, 0, params->get_int("nb_featuremap_pooling4") * sizeof(float), h_conv3_pooling4_coefficient, NULL, &event);

	int size_of_h_pooling4_conv5_synapses;
	if(in_has_bias) 
		size_of_h_pooling4_conv5_synapses = (params->get_int("nb_featuremap_conv5") * params->get_int("nb_featuremap_pooling4") * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + params->get_int("nb_featuremap_conv5")); // the last part is for biases or hessian of each featuremap in conv5

  	for (int fo = 0; fo < conv5_neurons.size(); fo++)
  	{
    		for (int fi = 0; fi < pooling4_neurons.size(); fi++) {
      			for (int fj = 0; fj < params->get_int("size_y_conv_kernel"); fj++)
				for (int fk = 0; fk < params->get_int("size_x_conv_kernel"); fk++) {
      				  if( pooling4_conv5_connection.at(fo).at(fi) == true)
					h_pooling4_conv5_synapses_values[fo * pooling4_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] =  pooling4_conv5_synapses.at(fo).at(fi)->values.at(fj).at(fk);
				  else
					h_pooling4_conv5_synapses_values[fo * pooling4_neurons.size() * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fi * params->get_int("size_y_conv_kernel") * params->get_int("size_x_conv_kernel") + fj * params->get_int("size_x_conv_kernel") + fk] = 0;
				}
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < conv5_neurons.size(); fo++) {
			// the organization is : [ nb_featuremap_conv5 * nb_featuremap_pooling4 * size_y_conv_kernel * size_x_conv_kernel + nb_featuremap_conv5] 
			h_pooling4_conv5_synapses_values[(size_of_h_pooling4_conv5_synapses - conv5_neurons.size()) + fo] = pooling4_conv5.at(fo)->bias_weight; 
		}
	}

	opencl_err = queue.enqueueWriteBuffer(d_pooling4_conv5_synapses_values, CL_TRUE, 0, size_of_h_pooling4_conv5_synapses * sizeof(float), h_pooling4_conv5_synapses_values, NULL, &event);

	int size_of_h_conv5_hidden6_synapses; 
	if(in_has_bias) 
		size_of_h_conv5_hidden6_synapses = (params->get_int("nb_neuron_hidden6") * params->get_int("nb_featuremap_conv5") + params->get_int("nb_neuron_hidden6")); // the last part is for biases of each neuron in hidden6
  	for (int fo = 0; fo < params->get_int("nb_neuron_hidden6"); fo++)
  	{
    		for (int fi = 0; fi < conv5_neurons.size(); fi++) {
			h_conv5_hidden6_synapses_values[fo * conv5_neurons.size() + fi] =  conv5_hidden6_synapse->values.at(fo).at(fi);
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < params->get_int("nb_neuron_hidden6"); fo++) {
			// the organization is : [ nb_neuron_hidden6 * nb_featuremap_conv5 * size_y_conv_kernel * size_x_conv_kernel + nb_neuron_hidden6] 
			h_conv5_hidden6_synapses_values[(size_of_h_conv5_hidden6_synapses - params->get_int("nb_neuron_hidden6")) + fo] = conv5_hidden6_synapse->values.at(fo).at(conv5_neurons.size());
		}
	}

	opencl_err = queue.enqueueWriteBuffer(d_conv5_hidden6_synapses_values, CL_TRUE, 0, size_of_h_conv5_hidden6_synapses * sizeof(float), h_conv5_hidden6_synapses_values, NULL, &event);

	int size_of_h_hidden6_output_synapses; 
	if(in_has_bias) 
		size_of_h_hidden6_output_synapses = (params->get_int("nb_neuron_output") * params->get_int("nb_neuron_hidden6") + params->get_int("nb_neuron_output")); // the last part is for biases of each neuron in output
  	for (int fo = 0; fo < params->get_int("nb_neuron_output"); fo++)
  	{
    		for (int fi = 0; fi < params->get_int("nb_neuron_hidden6"); fi++) {
			h_hidden6_output_synapses_values[fo * params->get_int("nb_neuron_hidden6") + fi] =  hidden6_output_synapse->values.at(fo).at(fi);
    		}
  	}

	if(in_has_bias) {
  		for (int fo = 0; fo < params->get_int("nb_neuron_output"); fo++) {
			// the organization is : [ nb_neuron_output * nb_neuron_hidden6 * size_y_conv_kernel * size_x_conv_kernel + nb_neuron_output] 
			h_hidden6_output_synapses_values[(size_of_h_hidden6_output_synapses - params->get_int("nb_neuron_output")) + fo] = hidden6_output_synapse->values.at(fo).at(params->get_int("nb_neuron_hidden6"));
		}
	}

	opencl_err = queue.enqueueWriteBuffer(d_hidden6_output_synapses_values, CL_TRUE, 0, size_of_h_hidden6_output_synapses * sizeof(float), h_hidden6_output_synapses_values, NULL, &event);


	queue.finish();
  }
}

void lenet5::forward(bool _backpropagation)
{
  if (dbg) cout << "--- begin forward ---" << endl;
  if (dbg) cout << "---- input neurons ----" << endl;
  if (dbg) 
  {
    cout << "Begin dump input neurons:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_input"); fo++)
      {
        cout << "Input neurons: " << " Inputs to " << "Inputs.dump" << endl;
        ofstream file;
        file.open("Inputs.dump");
        file << input_neurons.at(0)->dump();
        file.close();   
      }
  }

  if (dbg) cout << "---- input_conv1 ----" << endl;
  // forward input to conv1
  for (int iy = 0, oy = 0; iy + params->get_int("size_y_conv_kernel") <=  params->get_int("size_y_input"); iy += params->get_int("step_y_conv"), oy++)
    for (int ix = 0, ox = 0; ix + params->get_int("size_x_conv_kernel") <=  params->get_int("size_x_input"); ix += params->get_int("step_x_conv"), ox++)
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
        input_conv1.at(fo)->forward_prop(ix, iy, ox, oy, _backpropagation);
  if (dbg) 
  {
    cout << "Begin dump input_conv1:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
      {
        cout << "subnet: " << input_conv1.at(fo)->name << " to " << input_conv1.at(fo)->name + ".dump" << endl;
        ofstream file;
	file.open((input_conv1.at(fo)->name + ".dump").c_str());
        input_conv1.at(fo)->dump(file);
        file.close();   
      }
  }
  
  if (dbg) cout << "---- conv1_pooling2 ----" << endl;
  // forward conv1 to pooling2
  for (int iy = 0, oy = 0; iy + params->get_int("size_y_pooling_kernel") <=  params->get_int("size_y_conv1"); iy += params->get_int("size_y_pooling_kernel"), oy++)
    for (int ix = 0, ox = 0; ix + params->get_int("size_x_pooling_kernel") <=  params->get_int("size_x_conv1"); ix += params->get_int("size_x_pooling_kernel"), ox++)
      for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
        conv1_pooling2.at(fo)->forward_prop(ix, iy, ox, oy, _backpropagation);
  if (dbg) 
  {
    cout << "Begin dump conv1_pooling2:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
      {
        cout << "subnet: " << conv1_pooling2.at(fo)->name << " to " << conv1_pooling2.at(fo)->name + ".dump" << endl;
        ofstream file;
	file.open((conv1_pooling2.at(fo)->name + ".dump").c_str());
        conv1_pooling2.at(fo)->dump(file);
        file.close();   
      }
  }

  if (dbg) cout << "---- pooling2_conv3 ----" << endl;
  // forward pooling2 to conv3
  for (int iy = 0, oy = 0; iy + params->get_int("size_y_conv_kernel") <=  params->get_int("size_y_pooling2"); iy += params->get_int("step_y_conv"), oy++)
    for (int ix = 0, ox = 0; ix + params->get_int("size_x_conv_kernel") <=  params->get_int("size_x_pooling2"); ix += params->get_int("step_x_conv"), ox++)
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
        pooling2_conv3.at(fo)->forward_prop(ix, iy, ox, oy, _backpropagation);
  if (dbg) 
  {
    cout << "Begin dump pooling2_conv3:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      {
        cout << "subnet: " << pooling2_conv3.at(fo)->name << " to " << pooling2_conv3.at(fo)->name + ".dump" << endl;
        ofstream file;
	file.open((pooling2_conv3.at(fo)->name + ".dump").c_str());
        pooling2_conv3.at(fo)->dump(file);
        file.close();   
      }
  }

  if (dbg) cout << "---- conv3_pooling4 ----" << endl;
  // forward conv3 to pooling4
  for (int iy = 0, oy = 0; iy + params->get_int("size_y_pooling_kernel") <=  params->get_int("size_y_conv3"); iy += params->get_int("size_y_pooling_kernel"), oy++)
    for (int ix = 0, ox = 0; ix + params->get_int("size_x_pooling_kernel") <=  params->get_int("size_x_conv3"); ix += params->get_int("size_x_pooling_kernel"), ox++)
      for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
        conv3_pooling4.at(fo)->forward_prop(ix, iy, ox, oy, _backpropagation);
  if (dbg) 
  {
    cout << "Begin dump conv3_pooling4:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
      {
        cout << "subnet: " << conv3_pooling4.at(fo)->name << " to " << conv3_pooling4.at(fo)->name + ".dump" << endl;
        ofstream file;
	file.open((conv3_pooling4.at(fo)->name + ".dump").c_str());
        conv3_pooling4.at(fo)->dump(file);
        file.close();   
      }
  }

  if (dbg) cout << "---- pooling4_conv5 ----" << endl;
  // forward pooling4 to conv5
  for (int iy = 0, oy = 0; iy + params->get_int("size_y_conv_kernel") <=  params->get_int("size_y_pooling4"); iy += params->get_int("step_y_conv"), oy++)
    for (int ix = 0, ox = 0; ix + params->get_int("size_x_conv_kernel") <=  params->get_int("size_x_pooling4"); ix += params->get_int("step_x_conv"), ox++)
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
        pooling4_conv5.at(fo)->forward_prop(ix, iy, ox, oy, _backpropagation);
  if (dbg) 
  {
    cout << "Begin dump pooling4_conv5:" << endl;
      for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      {
        cout << "subnet: " << pooling4_conv5.at(fo)->name << " to " << pooling4_conv5.at(fo)->name + ".dump" << endl;
        ofstream file;
	file.open((pooling4_conv5.at(fo)->name + ".dump").c_str());
        pooling4_conv5.at(fo)->dump(file);
        file.close();   
      }
  }

  if (dbg) cout << "---- conv5_hidden6 ----" << endl;
  // forward conv5 to hidden6
  conv5_hidden6->mcp(_backpropagation);
  if (dbg) 
  {
    cout << "Begin dump conv5_hidden6:" << endl;
    cout << "subnet: " << "conv5_hidden6" << " to " << "conv5_hidden6.dump" << endl;
    ofstream file;
    file.open("conv5_hidden6.dump");
    conv5_hidden6->dump(file);
    file.close();   
  }

  // if (dbg) hidden_neurons->dump();
  if (dbg) cout << "---- hidden6_output ----" << endl;
  hidden6_output->mcp(_backpropagation);
  if (dbg) 
  {
    cout << "Begin dump hidden6_output:" << endl;
    cout << "subnet: " << "hidden6_output" << " to " << "hidden6_output.dump" << endl;
    ofstream file;
    file.open("hidden6_output.dump");
    hidden6_output->dump(file);
    file.close();   
  }

  //if (dbg) output_neurons->dump();
  if (dbg) cout << "--- end forward ---" << endl;
}

// train on a data set;
float lenet5::train(int _nb_epochs, data_set_mnist* _train, bool _use_second_order) 
{
  float error;
  float learning_rate_tmp = params->get_float("learning_rate");

  if(_use_second_order) hessian_estimation(_train);
  for (int epoch = 1; epoch <= _nb_epochs; epoch++) 
  {
    // shuffle input patterns at each epoch;
    //TODO maybe no need to do shuffle so often ? (time consuming);
    //TODO add shuffle() for class data_set_mnist
    // _train->shuffle();
    //cout << "dbg WARNING: not shuffling data" << endl;
    cout << "epoch "<< epoch << " starts" << endl;
    cout << "global learning rate: " << learning_rate_tmp << endl;
    // call backpropagation, and get error for epoch; 
    _train->shuffle();
    // Add on 11/18 by junjie
    error = train_back_propagation(_train, _use_second_order);
    // dump error every N epochs;
    //      if (verbose)
    //if (epoch % 100 == 0) cout << "--- epoch=" << epoch << ", error=" << error << " ---" << endl;
    cout << "epoch " << epoch << " ends, " << " MSE: " << error << endl;

    if(!(epoch%epochs_for_hessian_estimation) && _use_second_order && epoch!=_nb_epochs)
      hessian_estimation(_train);
    // adjust the learning rate according to the epochs from big to small
    if(!(epoch%2) && learning_rate_tmp>LR_threshold)
    {
      learning_rate_tmp = learning_rate_tmp*0.8;
      if(learning_rate_tmp < LR_threshold) learning_rate_tmp = LR_threshold;
      params->set_float("learning_rate",learning_rate_tmp);
    }

  } // for epoch
  //cout << "error=" << error << endl;
  return error;
}

// back-propagation;
float lenet5::train_back_propagation(data_set_mnist* train, bool _use_second_order) 
{
  verbose = false;
  float mse = 0;
  float mis_count = 0;
  //cout << "#rows: " << train->get_size() << " //mlp::train_back_prop" << endl;
  for (int i = 0; i < train->get_size(); i++) 
  {
    /*
    if(!((i+1)%10000))
    {
      cout << i+1 <<" in 60000" << endl;
    }
    */
    //cout << "dbg WARNING: using only one data_row_mnist" << endl;
    //for (int i = 0; i < 1; i++) 
    //{
    //cout << "--- Normalize image pixels---" << endl;  
    //train->normalize();
    data_row_mnist* row = train->rows.at(i);
    if (verbose)
    { 
      cout << "row: " << i << " //mlp::train_back_propagation" << endl;
      row->dump();
    }

    input_neurons.at(0)->set(row->inputs);
    // compute all the node outputs for this row;
    //

    forward(true);

    if(judgement() != row->label) mis_count++;
    // dump output and expected values;
    if (verbose) 
      output_neurons->dump(row->outputs);
    // compute error for this data entry; 
    mse += output_neurons->mse(row->outputs);
    if (dbg) cout << "dbg mse: " << mse << endl;

    // - back-propagation -
    if (dbg) cout << "--- hidden6_output ---" << endl;	
    // compute gradients in output layer;
    hidden6_output->compute_gradients_out(row->outputs);
    // compute gradients in hidden6 layer;
    hidden6_output->compute_gradients_in();
    // update weights between output and hidden6 layers;
    hidden6_output->update_weights(_use_second_order);

    if (dbg) cout << "--- conv5_hidden6 ---" << endl;
    // compute gradients in conv5 layer;
    conv5_hidden6->compute_gradients_in();
    // update weights between hidden6 and conv5 layers;
    conv5_hidden6->update_weights(_use_second_order);

    // Must compute_gradients_in BEFORE update_weights in convolutional subnets;
    // compute gradients in pooling4 layer;
    if (dbg) cout << "--- pooling4_conv5 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      pooling4_conv5.at(fo)->compute_gradients_in();
    // update weights between conv5 and pooling4 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      pooling4_conv5.at(fo)->update_weights(_use_second_order);  

    if (dbg) cout << "--- conv3_pooling4 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
    {
    // compute gradients in conv3 layer;
      conv3_pooling4.at(fo)->compute_gradients_in();
    // update weights between pooling4 and conv3 layers;
      conv3_pooling4.at(fo)->update_weights(_use_second_order);  
    }

    if (dbg) cout << "--- pooling2_conv3 ---" << endl;
    // compute gradients in pooling2 layer;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      pooling2_conv3.at(fo)->compute_gradients_in();
    // update weights between conv3 and pooling2 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      pooling2_conv3.at(fo)->update_weights(_use_second_order);  

    if (dbg) cout << "--- conv1_pooling2 ---" << endl;
    // then compute gradients in conv1 layer;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
    {
      conv1_pooling2.at(fo)->compute_gradients_in();
    // update weights between pooling4 and conv3 layers;
      conv1_pooling2.at(fo)->update_weights(_use_second_order);  
    }

    if (dbg) cout << "--- input_conv1 ---" << endl;
    // update weights between input and conv1 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
      input_conv1.at(fo)->update_weights(_use_second_order); 
  } // for data entries

  // mean squared error
  // mse = mse;// / (float)(train->get_size());
  cout << "mis classification: " << mis_count << endl;

  return mse;
}

float lenet5::test_mnist(data_set_mnist* _test)
{
  float correct = 0;
  data_row_mnist* row;

  cout << "number of testing img: " << _test->get_size() << endl;

  for(int i=0; i< _test->get_size(); i++)
  {
    row = _test->rows.at(i);
    input_neurons.at(0)->set(row->inputs);

    forward(false);
    if(judgement() == row->label) correct++;
    else
    {
	ofstream file;
	file.open("judgement.dump", ios::app);
	file << "iteration  " << i << endl;
	for(int j=0; j<output_neurons->size; j++)
          file << output_neurons->values.at(j) << " ";
        file << "judge: " << judgement() << " label: " << row->label << endl;
        file.close();
    }   

  }

  float err_rate = 1 - correct/(float)_test->get_size();
  cout << "the error rate is:" << err_rate << endl;
  cout<< "Test finish! seeya!" << endl;
  return err_rate; 
}

int lenet5::judgement()
{
  float max = -10000;
  int label=0;
  int length = (output_neurons->has_bias)? output_neurons->size-1:output_neurons->size;
  for(int i=0; i<length; i++)
  {
    if(output_neurons->values.at(i)>max)
    {
      max = output_neurons->values.at(i);
      label = i;
    }
  }
  return label;
}

void lenet5::hessian_estimation(data_set_mnist* train)
{
  cout << "commencing computing hessian information" << endl;
  clear_hessian_information();
  for(int i=0; i<nb_sampled_patterns; i++)
  {
    int rank = rand() % train->rows.size();
    data_row_mnist* row = train->rows.at(rank);
    input_neurons.at(0)->set(row->inputs);
    // compute all the node outputs for this row;
    forward(true);

    if (dbg) cout << "--- hidden6_output ---" << endl; 
    // compute gradients in output layer;
    hidden6_output->compute_second_gradients_out();
    // compute gradients in hidden6 layer;
    hidden6_output->compute_second_gradients_in();
    // update weights between output and hidden6 layers;
    hidden6_output->update_hessian(nb_sampled_patterns);

    if (dbg) cout << "--- conv5_hidden6 ---" << endl;
    // compute gradients in conv5 layer;
    conv5_hidden6->compute_second_gradients_in();
    // update weights between hidden6 and conv5 layers;
    conv5_hidden6->update_hessian(nb_sampled_patterns);

    // Must compute_gradients_in BEFORE update_weights in convolutional subnets;
    // compute gradients in pooling4 layer;
    if (dbg) cout << "--- pooling4_conv5 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      pooling4_conv5.at(fo)->compute_second_gradients_in();
    // update weights between conv5 and pooling4 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      pooling4_conv5.at(fo)->update_hessian(nb_sampled_patterns);  

    if (dbg) cout << "--- conv3_pooling4 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
    {
    // compute gradients in conv3 layer;
      conv3_pooling4.at(fo)->compute_second_gradients_in();
    // update weights between pooling4 and conv3 layers;
      conv3_pooling4.at(fo)->update_hessian(nb_sampled_patterns);  
    }

    if (dbg) cout << "--- pooling2_conv3 ---" << endl;
    // compute gradients in pooling2 layer;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      pooling2_conv3.at(fo)->compute_second_gradients_in();
    // update weights between conv3 and pooling2 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      pooling2_conv3.at(fo)->update_hessian(nb_sampled_patterns);  

    if (dbg) cout << "--- conv1_pooling2 ---" << endl;
    // then compute gradients in conv1 layer;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
    {
      conv1_pooling2.at(fo)->compute_second_gradients_in();
    // update weights between pooling4 and conv3 layers;
      conv1_pooling2.at(fo)->update_hessian(nb_sampled_patterns);  
    }

    if (dbg) cout << "--- input_conv1 ---" << endl;
    // update weights between input and conv1 layers;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
      input_conv1.at(fo)->update_hessian(nb_sampled_patterns); 
  }
}

void lenet5::clear_hessian_information()
{
   if (dbg) cout << "--- hidden6_output ---" << endl; 
    hidden6_output->clear_hessian_information();

    if (dbg) cout << "--- conv5_hidden6 ---" << endl;
    conv5_hidden6->clear_hessian_information();

    if (dbg) cout << "--- pooling4_conv5 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv5"); fo++)
      pooling4_conv5.at(fo)->clear_hessian_information();

    if (dbg) cout << "--- conv3_pooling4 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling4"); fo++)
      conv3_pooling4.at(fo)->clear_hessian_information();

    if (dbg) cout << "--- pooling2_conv3 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv3"); fo++)
      pooling2_conv3.at(fo)->clear_hessian_information();

    if (dbg) cout << "--- conv1_pooling2 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_pooling2"); fo++)
      conv1_pooling2.at(fo)->clear_hessian_information();

    if (dbg) cout << "--- input_conv1 ---" << endl;
    for (int fo = 0; fo < params->get_int("nb_featuremap_conv1"); fo++)
      input_conv1.at(fo)->clear_hessian_information(); 
}


