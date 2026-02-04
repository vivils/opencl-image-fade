#define CL_HPP_TARGET_OPENCL_VERSION 300


#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

struct ImgStruct {
    std::vector<float> data;
    int h;
    int w;
    int c;
};


const char* OpenCLSource = 
"__kernel void VectorMix(__global float* ratio, __global float* outIm, __global float* im1, __global float* im2)\n"
"{\n"
"    unsigned int n = get_global_id(0);\n"
"    float r = ratio[0];\n"
"    outIm[n] = r * im1[n] + (1.0f - r) * im2[n];\n"
"}\n";

int computeGPU(float ratio, ImgStruct *im1, ImgStruct *im2, ImgStruct *outIm) {
    size_t arrSize = im1->data.size();
    
    // Setup platform/device/context/queue (as in your code)
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create ratio buffer (single float)
    cl_mem buf_ratio = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      sizeof(float), &ratio, NULL);
    cl_mem buf_im1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    arrSize * sizeof(float), im1->data.data(), NULL);
    cl_mem buf_im2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    arrSize * sizeof(float), im2->data.data(), NULL);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                    arrSize * sizeof(float), NULL, NULL);

    // Build program (WITH ERROR CHECKING!)
    cl_program program = clCreateProgramWithSource(context, 1, &OpenCLSource, NULL, NULL);
    cl_int buildErr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (buildErr != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
        fprintf(stderr, "OpenCL Build Error:\n%s\n", log.data());
        return -1;
    }

    // Create kernel & SET ARGS IN CORRECT ORDER
    cl_kernel kernel = clCreateKernel(program, "VectorMix", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_ratio);  // ratio (float*)
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);    // outIm
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_im1);    // im1
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_im2);    // im2

    // Execute
    size_t globalSize = arrSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);  // Ensure completion before reading

    // Read result
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, arrSize * sizeof(float), 
                        outIm->data.data(), 0, NULL, NULL);

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buf_ratio);
    clReleaseMemObject(buf_im1);
    clReleaseMemObject(buf_im2);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}















//const char* OpenCLSource = 
//"__kernel void VectorMix(__global float* ratio, __global float* outIm, __global float* im1,__global float* im2)\n"
//"{\n"
//"   unsigned int n = get_global_id(0);\n"
//"   outIm[n] = ratio[0]*im1[n] + (1.f-ratio[0])*im2[n];\n"
//"}\n";
//
//int computeGPU(float ratio, ImgStruct *im1, ImgStruct *im2, ImgStruct *outIm) {
//    size_t arrSize = im1->data.size();
//    if (arrSize == 0) return -1;
//
//    // Get platform/device
//    cl_platform_id platform;
//    clGetPlatformIDs(1, &platform, NULL);
//    cl_device_id device;
//    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//
//    // Create context & command queue
//    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
//    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
//
//    // Create buffers (READ_ONLY for inputs, WRITE_ONLY for output)
//    cl_mem buf_im1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
//                                    arrSize * sizeof(float), im1->data.data(), NULL);
//    cl_mem buf_im2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
//                                    arrSize * sizeof(float), im2->data.data(), NULL);
//    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
//                                    arrSize * sizeof(float), NULL, NULL);
//
//    // Create & build program
//    cl_program program = clCreateProgramWithSource(context, 1, &OpenCLSource, NULL, NULL);
//    cl_int buildErr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
//    
//    // CRITICAL: Check for build errors!
//    if (buildErr != CL_SUCCESS) {
//        size_t logSize;
//        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
//        std::vector<char> log(logSize);
//        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
//        printf("OpenCL Build Error:\n%s\n", log.data());
//        return -1;
//    }
//
//    // Create kernel & set arguments IN CORRECT ORDER
//    cl_kernel kernel = clCreateKernel(program, "VectorAdd", NULL);
//    clSetKernelArg(kernel, 0, sizeof(float), &ratio);      // Scalar ratio
//    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_im1);   // im1
//    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_im2);   // im2
//    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_out);   // outIm
//
//    // Execute kernel
//    size_t globalSize = arrSize;
//    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
//    clFinish(queue); // Ensure kernel completes before reading
//
//    // Read result back
//    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, arrSize * sizeof(float), 
//                        outIm->data.data(), 0, NULL, NULL);
//
//    // Cleanup
//    clReleaseKernel(kernel);
//    clReleaseProgram(program);
//    clReleaseMemObject(buf_im1);
//    clReleaseMemObject(buf_im2);
//    clReleaseMemObject(buf_out);
//    clReleaseCommandQueue(queue);
//    clReleaseContext(context);
//
//    return 0;
//    //size_t arrSize = im1->data.size();
//    //float *imData1 = im1->data.data();
//    //float *imData2 = im2->data.data();
//    //float *outImData= outIm->data.data();
//
//    // cl_platform_id cpPlatform;
//
//    // clGetPlatformIDs(1, &cpPlatform, NULL);
//
//    // // Get a GPU device
//
//    // cl_device_id cdDevice;
//
//    // clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
//
//    // char cBuffer[1024];
//
//    // clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
//
//    // printf("CL_DEVICE_NAME: %s\n", cBuffer);
//
//    // clGetDeviceInfo(cdDevice, CL_DRIVER_VERSION, sizeof(cBuffer), &cBuffer, NULL);
//
//    // printf("CL_DRIVER_VERSION: %s\n\n", cBuffer);
//
//    // // Create a context to run OpenCL enabled GPU
//
//    // cl_context GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
//
//    // // Create a command-queue on the GPU device
//
//    // cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);
//
//    // // Allocate GPU memory for source vectors AND initialize from CPU memory
//
//
//    // cl_mem GPUVector1 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
//    // CL_MEM_COPY_HOST_PTR, sizeof(float) * arrSize, imData1, NULL);
//
//    // cl_mem GPUVector2 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
//    // CL_MEM_COPY_HOST_PTR, sizeof(float) * arrSize, imData2, NULL);
//
//    // cl_mem GPUOutputVector = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY,
//    // sizeof(float) * arrSize, NULL, NULL);
//
//    // cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 7, OpenCLSource, NULL, NULL);
//
//    // clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
//
//    // cl_kernel OpenCLVectorMix = clCreateKernel(OpenCLProgram, "VectorMix", NULL);
//
//
//    // clSetKernelArg(OpenCLVectorMix, 0, sizeof(float), &ratio);
//    // clSetKernelArg(OpenCLVectorMix, 1, sizeof(cl_mem), &GPUVector1);
//    // clSetKernelArg(OpenCLVectorMix, 2, sizeof(cl_mem), &GPUVector2);
//    // clSetKernelArg(OpenCLVectorMix, 3, sizeof(cl_mem), &GPUOutputVector);
//
//    // // Launch the Kernel on the GPU
//
//    // // This kernel only uses global data
//
//    // size_t WorkSize[1] = {arrSize}; // one dimensional Range
//
//    // clEnqueueNDRangeKernel(cqCommandQueue, OpenCLVectorMix, 1, NULL,
//
//    // WorkSize, NULL, 0, NULL, NULL);
//
//    // // Copy the output in GPU memory back to CPU memory
//
//    // clEnqueueReadBuffer(cqCommandQueue, GPUOutputVector, CL_TRUE, 0,
//
//    // arrSize * sizeof(float), outImData, 0, NULL, NULL);
//
//    // // Cleanup
//
//    // clReleaseKernel(OpenCLVectorMix);
//
//    // clReleaseProgram(OpenCLProgram);
//
//    // clReleaseCommandQueue(cqCommandQueue);
//
//    // clReleaseContext(GPUContext);
//
//    // clReleaseMemObject(GPUVector1);
//
//    // clReleaseMemObject(GPUVector2);
//
//    // clReleaseMemObject(GPUOutputVector);
//
//    // //for( int i =0 ; i < arrSize; i++)
//
//    // //     printf("[%d + %d = %d]\n",imData1[i], imData2[i], outImData[i]);
//
//    // return 0;
//
//}

int computeCPU(float ratio, ImgStruct *im1, ImgStruct *im2, ImgStruct *outIm) {
    float ratio2 = 1.f-ratio;
    size_t nPixels = im1->data.size();
    float *imData1 = im1->data.data();
    float *imData2 = im2->data.data();
    float *outData = outIm->data.data();
    for (size_t i = 0 ; i < nPixels ; i++){
        outData[i] = ratio*imData1[i] + ratio2*imData2[i];
    }
    return 0;
}

void copyMatDataToImStruct(cv::Mat *img, ImgStruct *imgStruct) {
    // CRITICAL: Must be float Mat (CV_32FC3) - verify before calling!
    assert(img->depth() == CV_32F && "Image must be float type!");
    
    std::vector<float> &vec = imgStruct->data;
    const int totalFloats = img->total() * img->channels(); // Total float elements

    if (img->isContinuous()) {
        // CORRECT: Cast uchar* to float* BEFORE pointer arithmetic
        vec.assign((float*)img->data, (float*)img->data + totalFloats);
    }
    imgStruct->h = img->rows;
    imgStruct->w = img->cols;
    imgStruct->c = img->channels();
}



cv::Mat copyImStructDataToMat(ImgStruct *imgStruct){
    int rows = imgStruct->h;
    int cols = imgStruct->w;
    int channels = imgStruct->c;
    int type = CV_32F;
    std::vector<float> *vec = &(imgStruct->data);
    cv::Mat img(rows, cols, type, vec->data());
    return img;
}

cv::Mat readImgStruct(char* imPath){
    cv::Mat image;
    image = cv::imread( imPath, cv::IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
    }
    return image;
}

void showImgStruct(ImgStruct *outIm, int type){
    cv::Mat frame(outIm->h, outIm->w, type, outIm->data.data());
    //out.write(frame);
 
    cv::imshow("Out image", frame);
    cv::waitKey(0);
    //// Display the frame
    cv::imshow("Frame", frame);
    if (cv::waitKey(1) == 'q') {
        return;
    }
}



int main(int argc, char** argv ) {
    if (argc != 3) {
        printf("usage: %s <image1> <image2>\n", argv[0]);
        return -1;
    }

    cv::Mat tmpImg1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat tmpImg2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    
    if (tmpImg1.empty() || tmpImg2.empty()) {
        printf("Failed to load images!\n");
        return -1;
    }

    cv::Mat image1, image2;
    tmpImg1.convertTo(image1, CV_32F, 1.0f / 255.0f); // Becomes CV_32FC3
    tmpImg2.convertTo(image2, CV_32F, 1.0f / 255.0f); // Becomes CV_32FC3

    std::cout << "Image1 type: " << image1.type() << " (CV_32FC" << image1.channels() << ")\n";
    std::cout << "Image2 type: " << image2.type() << " (CV_32FC" << image2.channels() << ")\n";

    // Convert to vector
    ImgStruct imStruct1, imStruct2;
    copyMatDataToImStruct(&image1, &imStruct1);
    copyMatDataToImStruct(&image2, &imStruct2);

    ImgStruct outIm;
    outIm.data.reserve(imStruct1.data.size());
    outIm.data.resize(imStruct1.data.size());
    outIm.h = imStruct1.h;
    outIm.w = imStruct1.w;
    outIm.c = imStruct1.c;

    // mix frame compute
    float ratio;
    int n = 100;
    const int type = CV_MAKETYPE(CV_32F, imStruct1.c); // Proper multi-channel type
    for (int i = 0 ; i < n ; i++){
        ratio = static_cast<float>(i) /n;
        std::cout << "Ratio: " << ratio << std::endl;
        int gpuRes = computeGPU(ratio, &imStruct1, &imStruct2, &outIm);
        //int cpuRes = computeCPU(ratio, &imStruct1, &imStruct2, &outIm);
        showImgStruct(&outIm, type);
    }
    cv::destroyAllWindows();
    return 0;
}
