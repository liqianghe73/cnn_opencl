
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../lib/net.hh"
#include "../lib/neurons.hh"
#include "../lib/synapses.hh"
#include "../lib/parameters.hh"
#include "../lib/data.hh"
#include "../lib/tools.hh"
#include "../input/data_mnist.hh"
#include "../subnets/mcp_bprop.hh"

#include "lenet5.hh"

#include "../subnets/conv_subnet3D_gpu.hh"
#include "../subnets/pooling_subnet2D_gpu.hh"
#include "../subnets/mcp_bprop_gpu.hh"
#include "../lib/subnet2D_gpu.hh"
#include "../subnets/gpu_device.hh"

using namespace std;

extern bool dbg;

void lenet5::forward_gpu(bool _backpropagation, int input_idx)
{
#if 0
#if GPU_COUNT_TIME
  float time;
  cudaEvent_t start,end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_conv_subnet3D_forward_prop_kernel(
		d_all_input_neurons, 
		input_idx,
		params->get_int("nb_featuremap_input"), 
		params->get_int("size_y_input"), 
		params->get_int("size_x_input"),
		d_input_conv1_synapses_values,
		in_has_bias,
		d_conv1_neurons,
		params->get_int("nb_featuremap_conv1"), 
		params->get_int("size_y_conv1"), 
		params->get_int("size_x_conv1"),
		params->get_int("size_y_conv_kernel"),
		params->get_int("size_x_conv_kernel"),
		params->get_int("step_y_conv"), 
		params->get_int("step_x_conv"),
		_backpropagation,
		d_input_conv1_derivatives_out,
		d_input_conv1_gradients_out,
		d_input_conv1_second_gradients_out
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_conv_forward_prop_kernel_input_conv1 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_pooling_subnet2D_forward_prop_kernel(
		d_pooling2_neurons, 
		params->get_int("nb_featuremap_pooling2"), 
		params->get_int("size_y_pooling2"), 
		params->get_int("size_x_pooling2"),
		params->get_int("size_y_pooling_kernel"),
		params->get_int("size_x_pooling_kernel"),
		d_conv1_pooling2_bias_weight,
		in_has_bias,
		d_conv1_pooling2_coefficient,
		d_conv1_neurons,
		params->get_int("size_y_conv1"),
		params->get_int("size_x_conv1"),
		'M',
		_backpropagation,
		d_conv1_pooling2_derivatives_out,
		d_conv1_pooling2_gradients_out,
		d_conv1_pooling2_second_gradients_out,
		d_conv1_pooling2_input_sampledown
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_pooling_forward_prop_kernel_conv1_pooling2 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_conv_subnet3D_forward_prop_kernel(
		d_pooling2_neurons, 
		0,	// the first 
		params->get_int("nb_featuremap_pooling2"), 
		params->get_int("size_y_pooling2"), 
		params->get_int("size_x_pooling2"),
		d_pooling2_conv3_synapses_values,
		in_has_bias,
		d_conv3_neurons,
		params->get_int("nb_featuremap_conv3"), 
		params->get_int("size_y_conv3"), 
		params->get_int("size_x_conv3"),
		params->get_int("size_y_conv_kernel"),
		params->get_int("size_x_conv_kernel"),
		params->get_int("step_y_conv"), 
		params->get_int("step_x_conv"),
		_backpropagation,
		d_pooling2_conv3_derivatives_out,
		d_pooling2_conv3_gradients_out,
		d_pooling2_conv3_second_gradients_out
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_conv_forward_prop_kernel_pooling2_conv3 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_pooling_subnet2D_forward_prop_kernel(
		d_pooling4_neurons, 
		params->get_int("nb_featuremap_pooling4"), 
		params->get_int("size_y_pooling4"), 
		params->get_int("size_x_pooling4"),
		params->get_int("size_y_pooling_kernel"),
		params->get_int("size_x_pooling_kernel"),
		d_conv3_pooling4_bias_weight,
		in_has_bias,
		d_conv3_pooling4_coefficient,
		d_conv3_neurons,
		params->get_int("size_y_conv3"),
		params->get_int("size_x_conv3"),
		'M',
		_backpropagation,
		d_conv3_pooling4_derivatives_out,
		d_conv3_pooling4_gradients_out,
		d_conv3_pooling4_second_gradients_out,
		d_conv3_pooling4_input_sampledown
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_pooling_forward_prop_kernel_conv3_pooling4 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_conv_subnet3D_forward_prop_kernel_p4c5(
		d_pooling4_neurons, 
		0,
		params->get_int("nb_featuremap_pooling4"), 
		params->get_int("size_y_pooling4"), 
		params->get_int("size_x_pooling4"),
		d_pooling4_conv5_synapses_values,
		in_has_bias,
		d_conv5_neurons,
		params->get_int("nb_featuremap_conv5"), 
		params->get_int("size_y_conv5"), 
		params->get_int("size_x_conv5"),
		params->get_int("size_y_conv_kernel"),
		params->get_int("size_x_conv_kernel"),
		params->get_int("step_y_conv"), 
		params->get_int("step_x_conv"),
		_backpropagation,
		d_pooling4_conv5_derivatives_out,
		d_pooling4_conv5_gradients_out,
		d_pooling4_conv5_second_gradients_out
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_conv_forward_prop_kernel_pooling4_conv5 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_mcp_forward_prop_kernel(
		d_hidden6_neurons, 
		0,
		params->get_int("nb_neuron_hidden6"), 
		d_conv5_hidden6_synapses_values,
		in_has_bias, 
		d_conv5_neurons,
		params->get_int("nb_featuremap_conv5"), 
		_backpropagation,
		d_conv5_hidden6_derivatives_out
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_mcp_kernel_conv5_hidden6 "  << time << endl;
#endif

#if GPU_COUNT_TIME
  cudaEventRecord(start, 0);
#endif

  call_mcp_forward_prop_kernel(
		d_all_output_neurons, 
		input_idx,
		params->get_int("nb_neuron_output"), 
		d_hidden6_output_synapses_values,
		in_has_bias, 
		d_hidden6_neurons,
		params->get_int("nb_neuron_hidden6"), 
		_backpropagation,
		d_hidden6_output_derivatives_out
		);

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT call_mcp_kernel_hidden6_output "  << time << endl;
#endif
#endif
}


// train on a data set;
float lenet5::train_gpu(int _nb_epochs, data_set_mnist* _train, bool _use_second_order) 
{
#if 0
  float error;
  float learning_rate_tmp = params->get_float("learning_rate");

  if(_use_second_order) hessian_estimation_gpu(_train);
  for (int epoch = 1; epoch <= _nb_epochs; epoch++) 
  //for (int epoch = 1; epoch <=1; epoch++) 
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
    error = train_back_propagation_gpu(_train, _use_second_order);
    // dump error every N epochs;
    //      if (verbose)
    //if (epoch % 100 == 0) cout << "--- epoch=" << epoch << ", error=" << error << " ---" << endl;
    cout << "epoch " << epoch << " ends"<< endl;

    if(!(epoch%epochs_for_hessian_estimation) && _use_second_order && epoch!=_nb_epochs)
      hessian_estimation_gpu(_train);
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
#endif
}

// back-propagation;
float lenet5::train_back_propagation_gpu(data_set_mnist* train, bool _use_second_order) 
{
#if 0
//#if GPU_COUNT_TIME
  float fptime, bptime, time;
  cudaEvent_t start,end, start1,end1, start2,end2;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&start1);
  cudaEventCreate(&end1);
  cudaEventCreate(&start2);
  cudaEventCreate(&end2);

  fptime = bptime = 0;

  cudaEventRecord(start, 0);
//#endif

  verbose = false;
  //float mse = 0;
  float mis_count = 0;

#if GPU_COUNT_TIME
  cudaEventRecord(start1, 0);
#endif

  int input_neuron_size = params->get_int("nb_featuremap_input") * params->get_int("size_y_input") * params->get_int("size_x_input");
  h_all_input_neurons = (float*)malloc(sizeof(float) * train->get_size() * input_neuron_size); 
  h_all_row_outputs = (float*)malloc(sizeof(float) * train->get_size() * params->get_int("nb_neuron_output") ); 
  h_all_output_neurons = (float*)malloc(sizeof(float) * train->get_size() * params->get_int("nb_neuron_output")* sizeof(float) ); 
  for(int i=0; i< train->get_size(); i++)
  {
    data_row_mnist* row = train->rows.at(i);
    input_neurons.at(0)->set(row->inputs);

    //copy data to host array
    for(int k=0; k<row->inputs.size(); k++) {
	for( int kk=0; kk<row->inputs.at(k).size(); kk++)
		h_all_input_neurons[i * input_neuron_size + k * row->inputs.at(0).size() + kk] = row->inputs.at(k).at(kk);
    }

    for(int k=0; k<row->outputs.size(); k++) {
	h_all_row_outputs[i * params->get_int("nb_neuron_output") + k] = row->outputs.at(k);
    }
  }
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_input_neurons, train->get_size() * input_neuron_size * sizeof(float)));
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_output_neurons, train->get_size() * params->get_int("nb_neuron_output") * sizeof(float)));
  //copy data from host to device
  cutilSafeCallNoSync(cudaMemcpyAsync(d_all_input_neurons, h_all_input_neurons, train->get_size() * input_neuron_size * sizeof(float), cudaMemcpyHostToDevice));

  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_row_outputs, train->get_size() * params->get_int("nb_neuron_output") * sizeof(float)));
  cutilSafeCallNoSync(cudaMemcpyAsync(d_all_row_outputs, h_all_row_outputs, (train->get_size() * params->get_int("nb_neuron_output") * sizeof(float)), cudaMemcpyHostToDevice));

#if GPU_COUNT_TIME
  cudaEventRecord(end1, 0);
  cudaEventSynchronize(end1);
  cudaEventElapsedTime(&time, start1, end1);
  cout << "T_STAT back_propagation_input_data_cy "  << time << endl;
#endif

  for (int i = 0; i < train->get_size(); i++) 
  //for (int i = 0; i < 10; i++) 
  {
#if GPU_COUNT_TIME
    cudaEventRecord(start1, 0);
#endif
    // compute all the node outputs for this row;

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    forward_gpu(true, i);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    fptime += time;
    //cout << "T_STAT forward_bp " << i << " " << time << endl;
#endif

    // - back-propagation -
#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_mcp_compute_gradients_out_kernel(
						d_all_output_neurons, 
						d_all_row_outputs,
						i,
						d_hidden6_output_derivatives_out,
						params->get_int("nb_neuron_output"),
						d_hidden6_output_gradients_out						
					  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_gradients_out_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_mcp_compute_gradients_in_kernel(
						params->get_int("nb_neuron_hidden6"),		// nin
						params->get_int("nb_neuron_output"),		// nout
						d_hidden6_output_gradients_out,			
						d_hidden6_output_synapses_values,	
						d_conv5_hidden6_derivatives_out,   // h_hidden6_output_derivatives_in
						d_conv5_hidden6_gradients_out	   // h_hidden6_output_gradients_in,
					);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_gradients_in_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_mcp_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_neuron_hidden6"),		// nin
						params->get_int("nb_neuron_output"),		// nout
						in_has_bias,
						d_hidden6_output_gradients_out,			
						d_hidden6_neurons,				// nin
						d_hidden6_output_synapses_hessian,
						d_hidden6_output_synapses_values	
				  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_update_weight_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_mcp_compute_gradients_in_kernel(
						params->get_int("nb_featuremap_conv5"),		// nin
						params->get_int("nb_neuron_hidden6"),		// nout
						d_conv5_hidden6_gradients_out,			
						d_conv5_hidden6_synapses_values,	
						d_pooling4_conv5_derivatives_out,   // h_conv5_hidden6_derivatives_in
						d_pooling4_conv5_gradients_out	   // h_conv5_hidden6_gradients_in,
					);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_gradients_in_c5h6_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_mcp_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_conv5"),		// nin
						params->get_int("nb_neuron_hidden6"),		// nout
						in_has_bias,
						d_conv5_hidden6_gradients_out,			
						d_conv5_neurons,				// nin
						d_conv5_hidden6_synapses_hessian,
						d_conv5_hidden6_synapses_values	
				  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_update_weight_c5h6_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_conv_subnet3D_compute_gradients_in_new_kernel(
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv5"),
						params->get_int("size_y_conv5"),					// fout->size_y
						params->get_int("size_x_conv5"),					// fout->size_x
						d_pooling4_conv5_synapses_values,
						d_pooling4_conv5_gradients_out,
						d_conv3_pooling4_derivatives_out,		// derivatives_in
						d_conv3_pooling4_gradients_out,		// gradients_in
						d_pooling4_conv5_fin_temp		// d_fin_temp
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_gradients_in_p4c5_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_conv_subnet3D_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv5"),
						params->get_int("size_y_conv5"),					// fout->size_y
						params->get_int("size_x_conv5"),					// fout->size_x
						in_has_bias,
						d_pooling4_conv5_gradients_kernel,
						d_pooling4_conv5_gradients_bias,
						d_pooling4_conv5_gradients_out,
						d_pooling4_neurons,				// nin
						0,
						d_pooling4_conv5_synapses_hessian,
						d_pooling4_conv5_synapses_values
				  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_update_weight_p4c5_p " << i << " " << time << endl;
#endif

    if (conv3_pooling4.at(0)->op == "A") {
    call_pooling_subnet2D_compute_gradients_in_kernel_A(
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),
						params->get_int("size_x_conv3"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv3_pooling4_coefficient,
						d_conv3_pooling4_gradients_out,
						d_pooling2_conv3_derivatives_out,		// derivatives_in
						d_pooling2_conv3_gradients_out,		// gradients_in
						params->get_int("smem_policy")
						);

    call_pooling_subnet2D_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						in_has_bias,
						d_conv3_pooling4_input_sampledown,
						d_conv3_pooling4_bias_weight,
						d_conv3_pooling4_bias_weight_hessian,
						d_conv3_pooling4_coefficient,
						d_conv3_pooling4_coefficient_hessian,
						d_conv3_pooling4_gradients_out
				  );
    }
    else if (conv3_pooling4.at(0)->op == "M") {

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_pooling_subnet2D_compute_gradients_in_kernel_M(
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv3_pooling4_input_sampledown,
						d_conv3_pooling4_gradients_out,
						d_pooling2_conv3_derivatives_out,		// derivatives_in
						d_pooling2_conv3_gradients_out		// gradients_in
						);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT pool_compute_gradients_in_c3p4_p " << i << " " << time << endl;
#endif
    }
    else
 	tools::error("Wrong operation for pooling layer");

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_conv_subnet3D_compute_gradients_in_new_kernel(
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),					// fout->size_y
						params->get_int("size_x_conv3"),					// fout->size_x
						d_pooling2_conv3_synapses_values,
						d_pooling2_conv3_gradients_out,
						d_conv1_pooling2_derivatives_out,		// derivatives_in
						d_conv1_pooling2_gradients_out,		// gradients_in
						d_pooling2_conv3_fin_temp		// d_fin_temp
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_gradients_in_p2c3_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_conv_subnet3D_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),					// fout->size_y
						params->get_int("size_x_conv3"),					// fout->size_x
						in_has_bias,
						d_pooling2_conv3_gradients_kernel,
						d_pooling2_conv3_gradients_bias,
						d_pooling2_conv3_gradients_out,
						d_pooling2_neurons,				// nin
						0,
						d_pooling2_conv3_synapses_hessian,
						d_pooling2_conv3_synapses_values
				  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_update_weight_p2c3_p " << i << " " << time << endl;
#endif

    if (conv1_pooling2.at(0)->op == "A") {
    call_pooling_subnet2D_compute_gradients_in_kernel_A(
						params->get_int("nb_featuremap_conv1"),
						params->get_int("size_y_conv1"),
						params->get_int("size_x_conv1"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv1_pooling2_coefficient,
						d_conv1_pooling2_gradients_out,
						d_input_conv1_derivatives_out,		// derivatives_in
						d_input_conv1_gradients_out,		// gradients_in
						params->get_int("smem_policy")
						);

    call_pooling_subnet2D_update_weights_kernel(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						in_has_bias,
						d_conv1_pooling2_input_sampledown,
						d_conv1_pooling2_bias_weight,
						d_conv1_pooling2_bias_weight_hessian,
						d_conv1_pooling2_coefficient,
						d_conv1_pooling2_coefficient_hessian,
						d_conv1_pooling2_gradients_out
				  );
    }
    else if (conv1_pooling2.at(0)->op == "M") {

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_pooling_subnet2D_compute_gradients_in_kernel_M(
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv1_pooling2_input_sampledown,
						d_conv1_pooling2_gradients_out,
						d_input_conv1_derivatives_out,		// derivatives_in
						d_input_conv1_gradients_out		// gradients_in
						);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT pool_compute_gradients_in_c1p2_p " << i << " " << time << endl;
#endif
    }
    else
 	tools::error("Wrong operation for pooling layer");

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif
    call_conv_subnet3D_update_weights_kernel_ic1(
						_use_second_order,
						params->get_float("mu"),
						params->get_float("learning_rate"),
						params->get_int("nb_featuremap_input"),
						params->get_int("size_y_input"),
						params->get_int("size_x_input"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv1"),
						params->get_int("size_y_conv1"),					// fout->size_y
						params->get_int("size_x_conv1"),					// fout->size_x
						in_has_bias,
						d_input_conv1_gradients_kernel,
						d_input_conv1_gradients_bias,
						d_input_conv1_gradients_out,
						d_all_input_neurons,				// nin
						i,
						d_input_conv1_synapses_hessian,
						d_input_conv1_synapses_values
				  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    bptime += time;
    //cout << "T_STAT conv_compute_update_weight_ic1_p " << i << " " << time << endl;


    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    cudaEventElapsedTime(&time, start1, end1);
    cout << "T_STAT train_back_it " << i << " " << time << endl;
#endif

  } // for data entries
  mis_count = judgement_gpu(train);
  cout << "mis classification: " << mis_count << endl;

  //cout << "fptime = " << fptime/10.0 << endl;
  //cout << "bptime = " << bptime/10.0 << endl;

  free(h_all_input_neurons);
  free(h_all_output_neurons);
  cutilSafeCallNoSync(cudaFree(d_all_input_neurons));
  cutilSafeCallNoSync(cudaFree(d_all_output_neurons));
  cutilSafeCallNoSync(cudaFree(d_all_row_outputs));

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT train_back_p " << time << endl;
#endif
#endif
  return mse;
}

float lenet5::test_mnist_gpu(data_set_mnist* _test)
{
#if 0
#if GPU_COUNT_TIME
  float time;
  cudaEvent_t start,end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
#endif

  int input_neuron_size = params->get_int("nb_featuremap_input") * params->get_int("size_y_input") * params->get_int("size_x_input");
  h_all_input_neurons = (float*)malloc(sizeof(float) * _test->get_size() * input_neuron_size); 
  h_all_output_neurons = (float*)malloc(sizeof(float) * _test->get_size() * params->get_int("nb_neuron_output") * sizeof(float)); 
  for(int i=0; i< _test->get_size(); i++)
  {
    data_row_mnist* row = _test->rows.at(i);
    input_neurons.at(0)->set(row->inputs);

    //copy data to host array
    for(int k=0; k<row->inputs.size(); k++) {
	for( int kk=0; kk<row->inputs.at(k).size(); kk++)
		h_all_input_neurons[i * input_neuron_size + k * row->inputs.at(0).size() + kk] = row->inputs.at(k).at(kk);
    }

  }
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_input_neurons, _test->get_size() * input_neuron_size * sizeof(float)));
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_output_neurons, _test->get_size() * params->get_int("nb_neuron_output") * sizeof(float)));
  //copy data from host to device
  cutilSafeCallNoSync(cudaMemcpyAsync(d_all_input_neurons, h_all_input_neurons, _test->get_size() * input_neuron_size * sizeof(float), cudaMemcpyHostToDevice));

  for(int i=0; i< _test->get_size(); i++)
  {
#if GPU_COUNT_TIME
    cudaEventRecord(start, 0);
#endif

    forward_gpu(false, i);

#if GPU_COUNT_TIME
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cout << "T_STAT forward_p " << i << " " << time << endl;
#endif
  }

  int mis_count = judgement_gpu(_test);

  //float err_rate = 1 - correct/(float)_test->get_size();
  float err_rate = mis_count/(float)_test->get_size();
  cout << "the error rate is:" << err_rate << endl;
  cout<< "Test finish! seeya!" << endl;

  free(h_all_input_neurons);
  free(h_all_output_neurons);
  cutilSafeCallNoSync(cudaFree(d_all_input_neurons));
  cutilSafeCallNoSync(cudaFree(d_all_output_neurons));
#endif
  float err_rate = 1; 
  return err_rate; 
}

void lenet5::hessian_estimation_gpu(data_set_mnist* train)
{
#if 0
#if GPU_COUNT_TIME
  float time;
  cudaEvent_t start,end, start1,end1, start2,end2;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&start1);
  cudaEventCreate(&end1);
  cudaEventCreate(&start2);
  cudaEventCreate(&end2);

  cudaEventRecord(start, 0);
#endif

  cout << "commencing computing hessian information" << endl;
  clear_hessian_information_gpu();

#if GPU_COUNT_TIME
  cudaEventRecord(start1, 0);
#endif

  int input_neuron_size = params->get_int("nb_featuremap_input") * params->get_int("size_y_input") * params->get_int("size_x_input");
  h_all_input_neurons = (float*)malloc(sizeof(float) * nb_sampled_patterns * input_neuron_size); 
  for(int i=0; i< nb_sampled_patterns; i++)
  {
    int rank = rand() % train->rows.size();
    data_row_mnist* row = train->rows.at(rank);
    input_neurons.at(0)->set(row->inputs);

    //copy data to host array
    for(int k=0; k<row->inputs.size(); k++) {
	for( int kk=0; kk<row->inputs.at(k).size(); kk++)
		h_all_input_neurons[i * input_neuron_size + k * row->inputs.at(0).size() + kk] = row->inputs.at(k).at(kk);
    }

  }
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_input_neurons, nb_sampled_patterns * input_neuron_size * sizeof(float)));
  cutilSafeCallNoSync(cudaMalloc((void **)&d_all_output_neurons, nb_sampled_patterns * params->get_int("nb_neuron_output") * sizeof(float)));
  //copy data from host to device
  cutilSafeCallNoSync(cudaMemcpyAsync(d_all_input_neurons, h_all_input_neurons, nb_sampled_patterns * input_neuron_size * sizeof(float), cudaMemcpyHostToDevice));

#if GPU_COUNT_TIME
  cudaEventRecord(end1, 0);
  cudaEventSynchronize(end1);
  cudaEventElapsedTime(&time, start1, end1);
  cout << "T_STAT hessian_estimation_input_data_cy "  << time << endl;
#endif

  for(int i=0; i<nb_sampled_patterns; i++)
  {
#if GPU_COUNT_TIME
    cudaEventRecord(start1, 0);
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    forward_gpu(true, i);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT forward_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_mcp_compute_second_gradients_out_kernel(
						d_hidden6_output_derivatives_out,
						params->get_int("nb_neuron_output"),
						d_hidden6_output_second_gradients_out						
					  );
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_second_gradients_out_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_mcp_compute_second_gradients_in_kernel(
						params->get_int("nb_neuron_hidden6"),		// nin
						params->get_int("nb_neuron_output"),		// nout
						d_hidden6_output_second_gradients_out,			
						d_hidden6_output_synapses_values,	
						d_conv5_hidden6_derivatives_out,   // h_hidden6_output_derivatives_in
						d_conv5_hidden6_second_gradients_out	   // h_hidden6_output_second_gradients_in,
					);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_second_gradients_in_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_mcp_compute_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_neuron_hidden6"),		// nin->values.size
						params->get_int("nb_neuron_output"),		// nout->values.size
						in_has_bias,
						d_hidden6_output_second_gradients_out,			
						d_hidden6_neurons,				// nin
						d_hidden6_output_synapses_hessian
					);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_update_hessian_h6o_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_mcp_compute_second_gradients_in_kernel(
						params->get_int("nb_featuremap_conv5"),		// nin
						params->get_int("nb_neuron_hidden6"),		// nout
						d_conv5_hidden6_second_gradients_out,			
						d_conv5_hidden6_synapses_values,	
						d_pooling4_conv5_derivatives_out,   // h_conv5_hidden6_derivatives_in
						d_pooling4_conv5_second_gradients_out	   // h_conv5_hidden6_second_gradients_in,
					);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_second_gradients_in_c5h6_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_mcp_compute_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_conv5"),		// nin->values.size
						params->get_int("nb_neuron_hidden6"),		// nout->values.size
						in_has_bias,
						d_conv5_hidden6_second_gradients_out,			
						d_conv5_neurons,				// nin
						d_conv5_hidden6_synapses_hessian
					);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT mcp_compute_update_hessian_c5h6_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_conv_subnet3D_compute_second_gradients_in_kernel(
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv5"),
						params->get_int("size_y_conv5"),					// fout->size_y
						params->get_int("size_x_conv5"),					// fout->size_x
						d_pooling4_conv5_synapses_values,
						d_pooling4_conv5_second_gradients_out,
						d_conv3_pooling4_derivatives_out,		// derivatives_in
						d_conv3_pooling4_second_gradients_out,		// gradients_in
						d_pooling4_conv5_fin_temp		// d_fin_temp
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_second_gradients_in_p4c5_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_conv_subnet3D_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv5"),
						params->get_int("size_y_conv5"),					// fout->size_y
						params->get_int("size_x_conv5"),					// fout->size_x
						in_has_bias,
						d_pooling4_conv5_second_gradients_out_sum,
						d_pooling4_conv5_second_gradients_out,
						d_pooling4_neurons,				// nin
						0,
						d_pooling4_conv5_synapses_hessian
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_update_hessian_p4c5_p " << i << " " << time << endl;
#endif

    if (conv3_pooling4.at(0)->op == "A") {
    call_pooling_subnet2D_compute_second_gradients_in_kernel_A(
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),
						params->get_int("size_x_conv3"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv3_pooling4_coefficient,
						d_conv3_pooling4_second_gradients_out,
						d_pooling2_conv3_derivatives_out,		// derivatives_in
						d_pooling2_conv3_second_gradients_out,		// gradients_in

						params->get_int("smem_policy")
						);

    call_pooling_subnet2D_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						in_has_bias,
						d_conv3_pooling4_input_sampledown,
						d_conv3_pooling4_bias_weight_hessian,
						d_conv3_pooling4_coefficient_hessian,
						d_conv3_pooling4_second_gradients_out
				  );
    }
    else if (conv3_pooling4.at(0)->op == "M") {

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_pooling_subnet2D_compute_second_gradients_in_kernel_M(
						params->get_int("nb_featuremap_pooling4"),
						params->get_int("size_y_pooling4"),
						params->get_int("size_x_pooling4"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv3_pooling4_input_sampledown,
						d_conv3_pooling4_second_gradients_out,
						d_pooling2_conv3_derivatives_out,		// derivatives_in
						d_pooling2_conv3_second_gradients_out		// gradients_in
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT pool_compute_second_gradients_in_c3p4_p " << i << " " << time << endl;
#endif

    }
    else
 	tools::error("Wrong operation for pooling layer");

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_conv_subnet3D_compute_second_gradients_in_kernel(
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),					// fout->size_y
						params->get_int("size_x_conv3"),					// fout->size_x
						d_pooling2_conv3_synapses_values,
						d_pooling2_conv3_second_gradients_out,
						d_conv1_pooling2_derivatives_out,		// derivatives_in
						d_conv1_pooling2_second_gradients_out,		// gradients_in
						d_pooling2_conv3_fin_temp		// d_fin_temp
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_second_gradients_in_p2c3_p " << i << " " << time << endl;
#endif

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_conv_subnet3D_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv3"),
						params->get_int("size_y_conv3"),					// fout->size_y
						params->get_int("size_x_conv3"),					// fout->size_x
						in_has_bias,
						d_pooling2_conv3_second_gradients_out_sum,
						d_pooling2_conv3_second_gradients_out,
						d_pooling2_neurons,				// nin
						0,
						d_pooling2_conv3_synapses_hessian
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_update_hessian_p2c3_p " << i << " " << time << endl;
#endif

    if (conv1_pooling2.at(0)->op == "A") {
    call_pooling_subnet2D_compute_second_gradients_in_kernel_A(
						params->get_int("nb_featuremap_conv1"),
						params->get_int("size_y_conv1"),
						params->get_int("size_x_conv1"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv1_pooling2_coefficient,
						d_conv1_pooling2_second_gradients_out,
						d_input_conv1_derivatives_out,		// derivatives_in
						d_input_conv1_second_gradients_out,		// gradients_in

						params->get_int("smem_policy")
						);

    call_pooling_subnet2D_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						in_has_bias,
						d_conv1_pooling2_input_sampledown,
						d_conv1_pooling2_bias_weight_hessian,
						d_conv1_pooling2_coefficient_hessian,
						d_conv1_pooling2_second_gradients_out
				  );
    }
    else if (conv1_pooling2.at(0)->op == "M") {

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_pooling_subnet2D_compute_second_gradients_in_kernel_M(
						params->get_int("nb_featuremap_pooling2"),
						params->get_int("size_y_pooling2"),
						params->get_int("size_x_pooling2"),
						params->get_int("size_y_pooling_kernel"),
						params->get_int("size_x_pooling_kernel"),
						d_conv1_pooling2_input_sampledown,
						d_conv1_pooling2_second_gradients_out,
						d_input_conv1_derivatives_out,		// derivatives_in
						d_input_conv1_second_gradients_out		// gradients_in
						);

#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT pool_compute_second_gradients_in_c1p2_p " << i << " " << time << endl;
#endif
    }
    else
 	tools::error("Wrong operation for pooling layer");

#if GPU_COUNT_TIME
    cudaEventRecord(start2, 0);
#endif

    call_conv_subnet3D_update_hessian_kernel(
						nb_sampled_patterns,
						params->get_int("nb_featuremap_input"),
						params->get_int("size_y_input"),
						params->get_int("size_x_input"),
						params->get_int("size_y_conv_kernel"),
						params->get_int("size_x_conv_kernel"),
						params->get_int("step_y_conv"),
						params->get_int("step_x_conv"),
						params->get_int("nb_featuremap_conv1"),
						params->get_int("size_y_conv1"),					// fout->size_y
						params->get_int("size_x_conv1"),					// fout->size_x
						in_has_bias,
						d_input_conv1_second_gradients_out_sum,
						d_input_conv1_second_gradients_out,
						d_all_input_neurons,				// nin
						i,
						d_input_conv1_synapses_hessian
						);
#if GPU_COUNT_TIME
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    cudaEventElapsedTime(&time, start2, end2);
    cout << "T_STAT conv_compute_update_hessian_ic1_p " << i << " " << time << endl;


    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    cudaEventElapsedTime(&time, start1, end1);
    cout << "T_STAT hessian_estimation_it " << i << " " << time << endl;
#endif
  }

  free(h_all_input_neurons);
  cutilSafeCallNoSync(cudaFree(d_all_input_neurons));
  cutilSafeCallNoSync(cudaFree(d_all_output_neurons));

#if GPU_COUNT_TIME
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cout << "T_STAT hessian_estimation " << time << endl;
#endif
#endif
}

void lenet5::clear_hessian_information_gpu()
{
#if 0
#if GPU_COUNT_TIME
        float time;
  	cudaEvent_t start,end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);
  	cudaEventRecord(start, 0);
#endif

	int nb_neuron_output = params->get_int("nb_neuron_output");
	int nb_neuron_hidden6 = params->get_int("nb_neuron_hidden6");
	int nb_featuremap_conv5 = params->get_int("nb_featuremap_conv5");
	int nb_featuremap_pooling4 = params->get_int("nb_featuremap_pooling4");
	int nb_featuremap_conv3 = params->get_int("nb_featuremap_conv3");
	int nb_featuremap_pooling2 = params->get_int("nb_featuremap_pooling2");
	int nb_featuremap_conv1 = params->get_int("nb_featuremap_conv1");
	int nb_featuremap_input = params->get_int("nb_featuremap_input");
	int size_y_conv_kernel = params->get_int("size_y_conv_kernel");
	int size_x_conv_kernel = params->get_int("size_x_conv_kernel");

 	call_clear_3d_hessian(d_hidden6_output_synapses_hessian, nb_neuron_hidden6, 1, 1, 0, nb_neuron_output, 1, 1);

 	call_clear_3d_hessian(d_conv5_hidden6_synapses_hessian, nb_featuremap_conv5, 1, 1, 0, nb_neuron_hidden6, 1, 1);
 
 	call_clear_3d_hessian(d_pooling4_conv5_synapses_hessian, nb_featuremap_pooling4, size_x_conv_kernel, size_y_conv_kernel, in_has_bias, nb_featuremap_conv5, 1, 1);

 	if(in_has_bias) {
 		call_clear_hessian(d_conv3_pooling4_bias_weight_hessian, nb_featuremap_pooling4);
 	}

 	call_clear_hessian(d_conv3_pooling4_coefficient_hessian, nb_featuremap_pooling4);

 	call_clear_3d_hessian(d_pooling2_conv3_synapses_hessian, nb_featuremap_pooling2, size_x_conv_kernel, size_y_conv_kernel, in_has_bias, nb_featuremap_conv3, 1, 1);

 	if(in_has_bias) {
 		call_clear_hessian(d_conv1_pooling2_bias_weight_hessian, nb_featuremap_pooling2);
 	}

 	call_clear_hessian(d_conv1_pooling2_coefficient_hessian, nb_featuremap_pooling2);

 	call_clear_3d_hessian(d_input_conv1_synapses_hessian, nb_featuremap_input, size_x_conv_kernel, size_y_conv_kernel, in_has_bias, nb_featuremap_conv1, 1, 1);

#if GPU_COUNT_TIME
  	cudaEventRecord(end, 0);
  	cudaEventSynchronize(end);
  	cudaEventElapsedTime(&time, start, end);
  	cout << "T_STAT clear_hessian_p " << time << endl;
#endif
#endif
}


int lenet5::judgement_gpu(data_set_mnist* dataset)
{
  float max;
  int label;
  int mis_count = 0;
  int idx;
  float mse =0;
#if 0
  cutilSafeCallNoSync(cudaMemcpyAsync(h_all_output_neurons, d_all_output_neurons, (dataset->get_size() * params->get_int("nb_neuron_output") * sizeof(float)), cudaMemcpyDeviceToHost));

  for (int i = 0; i < dataset->get_size(); i++) {
	data_row_mnist* row = dataset->rows.at(i);

  	idx = i * params->get_int("nb_neuron_output");
  	for (int fo = 0; fo < params->get_int("nb_neuron_output"); fo++) {
        	output_neurons->values.at(fo) = h_all_output_neurons[idx + fo];
  	}

  	int length = (output_neurons->has_bias)? output_neurons->size-1:output_neurons->size;
	label = 0; 
	max = -10000;
  	for(int i=0; i<length; i++)
  	{
    		if(output_neurons->values.at(i)>max)
    		{
      		max = output_neurons->values.at(i);
      		label = i;
    		}
  	}
	if (label != row->label) mis_count++;

    	mse += output_neurons->mse(row->outputs);
  }
#endif
  printf("mse = %.0f\n", mse);
  return mis_count;
}

