/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdlib>
#include <cstring>

#include "taso/ops.h"
#include "taso/cuda_helper.h"
using namespace taso;

Model::Model()
: isTraining(false), print_cost(false)
{
  //int* a = (int*) malloc(sizeof(int) * 8);
  checkCUDA(cudaSetDevice(0));
  checkCUDNN(cudnnCreate(&dnn));
  checkCUDA(cublasCreate(&blas));
  workSpaceSize = WORK_SPACE_SIZE;
  global_unique_id = 100;
  checkCUDA(cudaMalloc(&workSpace, workSpaceSize));
  // printf("handle.workSpace = 0x%x\n", workSpace);
  // create all descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&scaleTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  // allocate tensors for measuring performance
  checkCUDA(cudaMalloc(&inputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&biasPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&outputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&filterPtr, MAX_TENSOR_SIZE));
  // create tensors for batch norm
  checkCUDA(cudaMalloc(&scalePtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningVar, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveVar, MAX_TENSOR_SIZE));
  // create cuda events
  checkCUDA(cudaEventCreate(&startEvent));
  checkCUDA(cudaEventCreate(&endEvent));
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList)
{
    // These are dummy data for development
    // @TODO: change tensor shape is determined by user input
    const int batch_count = 1;
    const int in_channel = 3;
    const int in_height = 224;
    const int in_width = 224;
    float inData[batch_count][in_channel][in_height][in_width];     // host input data
    float outData[1][2048][7][7];    // host output data
    float* inData_d;         // cuda device input data
    float* outData_d;        // cuda device output data

    for (int i = 0; i < batch_count; i++) {
        for (int j = 0; j < in_channel; j++) {
            for (int k = 0; k < in_height; k++) {
                for (int l = 0; l < in_width; l++) {
                    // test with random input
                    inData[i][j][k][l] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 10;
                }
            }
        }
    }

    // A pointer for the input of first operator, and a pointer for the output of last operator
    inData_d = (float *)((opBaseList[0]->inputs[0]).data_ptr);
    outData_d = (float *)((opBaseList[opBaseList.size()-1]->outputs[0]).data_ptr);

    // memcpy form CPU to GPU
    checkCUDA(cudaMemcpy(inData_d, inData, sizeof(inData), cudaMemcpyHostToDevice));

    // DEBUG
    printf("N: %d\n", opBaseList[0]->inputs[0].dim[0]);
    printf("C: %d\n", opBaseList[0]->inputs[0].dim[1]);
    printf("H: %d\n", opBaseList[0]->inputs[0].dim[2]);
    printf("W: %d\n", opBaseList[0]->inputs[0].dim[3]);
    printf("opBaseList.size(): %lu\n", opBaseList.size());
    
    
    const int num_runs = 1;
    /*
    // warmup
    for (int times = 0; times < num_runs; times++) {
        for (int i = 0; i < opBaseList.size(); i++) {
            opBaseList[i]->forward();
        }
    }
    */

    // measure runtime
    // checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaEventRecord(startEvent));
    for (int times = 0; times < num_runs; times++) {
        for (int i = 0; i < opBaseList.size(); i++) {
            opBaseList[i]->forward();

            printf("input addr: %p | ouput addr: %p\n", (float *)(opBaseList[i]->inputs[0].data_ptr), (float *)(opBaseList[i]->outputs[0].data_ptr));
            
            //Tensor temp = opBaseList[i]->outputs[0];
            //float tempInData[temp.dim[0]][temp.dim[1]][temp.dim[2]][temp.dim[3]];
            //checkCUDA(cudaMemcpy(tempInData, (float *)(opBaseList[i]->inputs[0].data_ptr), sizeof(tempInData), cudaMemcpyDeviceToHost));
            //printf("inData[0][0][0][0] of %dth operator: %.5f", i, inData[0][0][0][0]);
        }
        
        // printf("output address: %p\n", outData_d);
        checkCUDA(cudaMemcpy(outData, outData_d, sizeof(outData), cudaMemcpyDeviceToHost));
        std::cout << "outData:" << endl;
        std::cout << "[";
        for (int i = 0; i < 1; i++) {
            std::cout << "[";
            for (int j = 0; j < 2048; j++) {
                std::cout << "[";
                for (int k = 0; k < 7; k++) {
                    std::cout << "[";
                    for (int l = 0; l < 7; l++) {
                        printf("%.6f ", outData[i][j][k][l]);
                    }
                    std::cout << "]" << endl;
                }
                std::cout << "]" << endl;
            }
            std::cout << "]" << endl;
        }
        std::cout << "]" << endl;
    }

    checkCUDA(cudaEventRecord(endEvent));
    checkCUDA(cudaEventSynchronize(endEvent));
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
    return milliseconds / num_runs;
}

/********** Added ************/
// TensorHandle is typedef of Tensor *
// TODO output dimension should be calculated (passed by argument...?)
TensorHandle Model::get_runtime_output(const int* dims, const DATATYPE* data, const std::vector<OpBase*> *opBaseList)
{
    TensorHandle result = new Tensor();

    const int batch_count = dims[0];
    const int in_channel = dims[1];
    const int in_height = dims[2];
    const int in_width = dims[3];

    float inData[batch_count][in_channel][in_height][in_width];     // host input data
    float outData[1][256][7][7];    // host output data
    float* inData_d;         // cuda device input data
    float* outData_d;        // cuda device output data

    const std::vector<OpBase*> oplist = *(opBaseList);

    /*
    unsigned int offset = 0;
    for (int i = 0; i < batch_count; i++) {
        for (int j = 0; j < in_channel; j++) {
            for (int k = 0; k < in_height; k++) {
                for (int l = 0; l < in_width; l++) {
                    inData[i][j][k][l] = *(data + offset);
                    offset += 1;
                    //printf("[%.5f] ", inData[i][j][k][l]);
                }
            }
        }
    }*/
    memcpy(inData, data, sizeof(data)); 

    //printf("size: %lu\n", sizeof(inData));
    //printf("input shape: %d, %d, %d, %d\n", batch_count, in_channel, in_height, in_width);

    // A pointer for the input of first operator, and a pointer for the output of last operator
    inData_d = (float *)((oplist[0]->inputs[0]).data_ptr);
    outData_d = (float *)((oplist[oplist.size()-1]->outputs[0]).data_ptr);
    
    // memcpy form CPU to GPU
    checkCUDA(cudaMemcpy(inData_d, inData, sizeof(inData), cudaMemcpyHostToDevice));

    /*
    // DEBUG
    printf("N: %d\n", opBaseList[0]->inputs[0].dim[0]);
    printf("C: %d\n", opBaseList[0]->inputs[0].dim[1]);
    printf("H: %d\n", opBaseList[0]->inputs[0].dim[2]);
    printf("W: %d\n", opBaseList[0]->inputs[0].dim[3]);
    printf("opBaseList.size(): %lu\n", opBaseList.size());
    */
    
    //checkCUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < oplist.size(); i++) {
        oplist[i]->forward();
        // printf("input addr: %p | ouput addr: %p\n", (float *)(opBaseList[i]->inputs[0].data_ptr), (float *)(opBaseList[i]->outputs[0].data_ptr));      
    }
        
    // printf("output address: %p\n", outData_d);
    checkCUDA(cudaMemcpy(outData, outData_d, sizeof(outData), cudaMemcpyDeviceToHost));    // TODO uncomment this
    result->data_ptr = outData;
    
    /*
    std::cout << "outData:" << endl;
    std::cout << "[";
    for (int i = 0; i < 1; i++) {
        std::cout << "[";
        for (int j = 0; j < 2048; j++) {
            std::cout << "[";
            for (int k = 0; k < 7; k++) {
                std::cout << "[";
                for (int l = 0; l < 7; l++) {
                    printf("%.8f ", outData[i][j][k][l]);
                }
                std::cout << "]" << endl;
            }
            std::cout << "]" << endl;
        }
        std::cout << "]" << endl;
    }
    std::cout << "]" << endl;
    */

    //checkCUDA(cudaFree(inData_d));
    //checkCUDA(cudaFree(outData_d));

    //checkCUDA(cudaEventRecord(endEvent));
    //checkCUDA(cudaEventSynchronize(endEvent));
    //float milliseconds;
    //cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
    //printf("cudaEvent time: %.5f\n", milliseconds);

    // TODO outData should be returned
    return result;
}

void* Model::allocate_memory(size_t size, const DATATYPE* data_initial)
{
  void* ptr;
  if (size == 0) {
    // Note: Special value for zero-sized tensor
    ptr = (void*) 0x1;
  } else {
    checkCUDA(cudaMalloc(&ptr, size));
  }
  if (data_initial != NULL) {
    checkCUDA(cudaMemcpy(ptr, data_initial, size, cudaMemcpyDefault));
  }
  return ptr;
}

bool Model::copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size)
{
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
  return true;
}
