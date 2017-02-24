#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../lib/parameters.hh"
#include "../lib/tools.hh"
#include "../input/data_mnist.hh"
#include "lenet5.hh"
#include <utility>

using namespace std;

bool dbg = false;

int main(int argc, const char *argv[]) 
{
  bool verbose = false;
  // float mse;

  if (argc <=2 )
  {
      cerr << "Usage: " << endl << "\texcnn <load/unload> <cpu/gpu> <kernel_path>" << endl;
      return 1;
  }
 
  string f_load = argv[1];
  string f_device = argv[2];
  string f_kernel = argv[3];

  srand ( time(NULL) );

  // --- CNN ---
  cout << "--- LeNet5 CNN ---" << endl;

  // create an CNN;
  parameters cnfg;

  cnfg.set_string("kernel_path", f_kernel);

  // add computing device flag
  if (f_device == "cpu")
  {
  	cnfg.set_int("use_gpu", 0);
  }
  else if (f_device == "gpu")
  {
  	cnfg.set_int("use_gpu", 1);
  }
  else
  {
 	cout << "no definition for " << argv[2] << endl;
	return 1;
  }

    // --- forworad-propagation ---

  cnfg.set_bool("in_has_bias", true);
  cnfg.set_int("smem_policy", 0);

  cnfg.set_int("size_x_conv_kernel", 5);
  cnfg.set_int("size_y_conv_kernel", 5);
  cnfg.set_int("size_x_pooling_kernel", 2);
  cnfg.set_int("size_y_pooling_kernel", 2);
  cnfg.set_int("step_x_conv", 1);
  cnfg.set_int("step_y_conv", 1);
  cnfg.set_int("nb_featuremap_input", 1);
  cnfg.set_int("size_x_input", 32);
  cnfg.set_int("size_y_input", 32);
  cnfg.set_int("nb_featuremap_conv1", 6);
  cnfg.set_int("size_x_conv1", 28);
  cnfg.set_int("size_y_conv1", 28);
  cnfg.set_int("nb_featuremap_pooling2", 6);
  cnfg.set_int("size_x_pooling2", 14);
  cnfg.set_int("size_y_pooling2", 14);
  cnfg.set_int("nb_featuremap_conv3", 16);
  cnfg.set_int("size_x_conv3", 10);
  cnfg.set_int("size_y_conv3", 10);
  cnfg.set_int("nb_featuremap_pooling4", 16);
  cnfg.set_int("size_x_pooling4", 5);
  cnfg.set_int("size_y_pooling4", 5);
  cnfg.set_int("nb_featuremap_conv5", 120); //120
  cnfg.set_int("size_x_conv5", 1);
  cnfg.set_int("size_y_conv5", 1);
  cnfg.set_int("nb_neuron_hidden6", 80); //80
  cnfg.set_int("nb_neuron_output", 10);
  cnfg.set_float("learning_rate", 0.001);
  cnfg.set_float("mu", 0.1);
  cnfg.set_int("metric", mse);
  cnfg.set_float("activation", 1);
  cnfg.set_int("epochs for hessian estimation", 1);
  cnfg.set_int("number of patterns for hessian estimation", 500);
  //  cnfg.set_float("momentum", 0.7);
  cout << "--- Configs Set ---" << endl;

  lenet5 nn(&cnfg);

  cout << "Loading training data with mnist database" << endl;
  data_set_mnist* train = new data_set_mnist("input/train-images.idx3-ubyte", "input/train-labels.idx1-ubyte");
  cout << "--- convert training data from 28X28 to 32X32---" << endl; 

  train->resize(false, 32,32);
  cout << "--- Normalize image pixels---" << endl;  
  // train->normalize();
  train->normalization();

  cout<< "Loading test data with mnist database" <<endl;
  data_set_mnist* test = new data_set_mnist("input/t10k-images.idx3-ubyte", "input/t10k-labels.idx1-ubyte");
  cout << "--- convert testing data from 28X28 to 32X32---" << endl;
  test->resize(false, 32, 32);
  cout << "--- Normalize image pixels---" << endl;  
  // test->normalize();
  test->normalization();

  dbg = false;

  if (f_load == "load") 
  {
    cout << "--- Start loading---" << endl;
    nn.load();
  }

  if (f_device == "cpu")
  {
  	nn.train(40, train, true);
  	//nn.train(1, train, false);
  	nn.test_mnist(test);
  }
  else if (f_device == "gpu")
  {
  	nn.train_gpu(40, train, true);
  	//nn.train_gpu(1, train, false);
  	//nn.test_mnist_gpu(test);
  }

  //nn.dump();
  delete(train);
  delete(test);

  return 0;
}
