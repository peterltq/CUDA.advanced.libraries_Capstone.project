#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <helper_string.h>

#define DEFAULT_INPUT_IMAGE_FILE "data/sloth.png"

#define CHECK_CUDNN(status)                                                       \
    if (status != CUDNN_STATUS_SUCCESS)                                           \
    {                                                                             \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                                       \
    }

// Helper to split the file name by '.'
std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = s.find(delimiter);

    while (end != std::string::npos)
    {
        tokens.push_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delimiter, start);
    }

    tokens.push_back(s.substr(start)); // Last token
    return tokens;
}

// Allows output vector in cout.
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); i++)
    {
        os << vec[i];
        if (i < vec.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Generate the output file name. 
// E.g. abc.png -> abc_processed.png
std::string genOutputFileName(std::string input_image_path) {
    std::vector tokens = split(input_image_path, '.');
    // std::cout << "tokens:" << tokens << std::endl;

    std::string output_image_path;
    if (tokens.size() == 1) 
    {
        std::cerr << "File extension is required in input file: " << input_image_path << std::endl;
        return "";
    } else 
    {
        for (int i = 0; i < tokens.size() - 1; ++i) 
        {
            if (i > 0) 
            {
                output_image_path += ".";
            }
            output_image_path += tokens[i];
        }
        output_image_path += "_processed.";
        // Add the extension back.
        output_image_path += tokens[tokens.size() - 1];
    }
    return output_image_path;
}

int main(int argc, char *argv[])
{
    // 1. Load input image using OpenCV
    char *inputPath = NULL;
    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input", &inputPath);
    }
    
    std::string input_image_path;
    if (NULL == inputPath) 
    {
        input_image_path = DEFAULT_INPUT_IMAGE_FILE;
    } else {
        input_image_path = inputPath;
    }

    std::cout << "Loading input image file:" << input_image_path << std::endl;
    cv::Mat h_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if (h_image.empty())
    {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        return -1;
    }

    // Convert to float and normalize to [0, 1]
    // 3-channel 32-bit floating-point matrix. The values is normailzed to between [0. 1]
    h_image.convertTo(h_image, CV_32FC3, 1.0 / 255.0);

    // 2. Get image dimensions
    int height = h_image.rows;
    int width = h_image.cols;
    int channels = h_image.channels(); // Should be 3 for RGB

    std::cout << "Input image: " << width << "x" << height
              << " with " << channels << " channels" << std::endl;

    // 3. cuDNN setup
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    // 4. Define convolution parameters
    int n = 1;        // Batch size
    int c = channels; // Input channels (3 for RGB)
    int h = height;   // Input height
    int w = width;    // Input width

    int k = 3;  // Number of output filters
    int kh = 3; // Kernel height
    int kw = 3; // Kernel width

    int pad_h = 1; // Padding height (same convolution)
    int pad_w = 1; // Padding width
    int str_h = 1; // Stride height
    int str_w = 1; // Stride width

    // 5. Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 6. Set tensor dimensions (NCHW format)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n, c, h, w));

    // 7. Set filter dimensions (KCHW format)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        k, c, kh, kw));

    // 8. Set convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, // Padding
        str_h, str_w, // Stride
        1, 1,         // Dilation (stretch factors for height and weight)
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // 9. Get output dimensions
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc,
        input_desc,
        filter_desc,
        &out_n, &out_c, &out_h, &out_w));

    std::cout << "Output dimensions: " << out_h << "x" << out_w
              << " with " << out_c << " channels" << std::endl;

    // 10. Set output tensor descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

    // 11. Convert host image to NCHW format (HWC -> CHW)
    // OpenCV stores as HWC (height, width, channels)
    // cuDNN expects NCHW (batch, channels, height, width)
    std::vector<float> h_input_nchw(n * c * h * w);

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                // Index in NCHW: (n * c + ch) * h * w + i * w + j
                // Index in HWC: i * w * c + j * c + ch
                h_input_nchw[ch * h * w + i * w + j] = h_image.at<cv::Vec3f>(i, j)[ch];
            }
        }
    }

    // 12. Allocate device memory
    size_t input_size = n * c * h * w * sizeof(float);
    size_t filter_size = k * c * kh * kw * sizeof(float);
    size_t output_size = out_n * out_c * out_h * out_w * sizeof(float);

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, output_size);

    // 13. Copy input to device
    cudaMemcpy(d_input, h_input_nchw.data(), input_size, cudaMemcpyHostToDevice);

    // 14. Create and copy filter (simple edge detection kernel)
    // Sobel-like kernel: detects horizontal edges
    std::vector<float> h_filter(k * c * kh * kw);
    // Simple edge detection filter (Sobel-like)
    std::vector<float> sobel_kernel = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1};

    // Apply the same kernel to all input channels and output filters
    for (int out_ch = 0; out_ch < k; out_ch++)
    {
        for (int in_ch = 0; in_ch < c; in_ch++)
        {
            for (int i = 0; i < kh * kw; i++)
            {
                // Index in KCHW format: (out_ch * c + in_ch) * kh * kw + i
                h_filter[(out_ch * c + in_ch) * kh * kw + i] = sobel_kernel[i];
            }
        }
    }

    cudaMemcpy(d_filter, h_filter.data(), filter_size, cudaMemcpyHostToDevice);

    // 15. Choose the best algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        1,
        &algo_count,
        &algo_perf));

    std::cout << "Selected algorithm: " << algo_perf.algo << std::endl;
    std::cout << "Workspace needed: " << algo_perf.memory << " bytes" << std::endl;

    // 16. Get workspace size and allocate
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo_perf.algo,
        &workspace_size));

    void *d_workspace = nullptr;
    if (workspace_size > 0)
    {
        cudaMalloc(&d_workspace, workspace_size);
    }

    // 17. Perform convolution
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        input_desc, d_input,
        filter_desc, d_filter,
        conv_desc,
        algo_perf.algo,
        d_workspace,
        workspace_size,
        &beta,
        output_desc, d_output));

    std::cout << "Convolution completed!" << std::endl;

    // 18. Copy output back to host
    std::vector<float> h_output_nchw(out_n * out_c * out_h * out_w);
    cudaMemcpy(h_output_nchw.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // 19. Convert from NCHW to HWC format (for OpenCV)
    cv::Mat h_output(out_h, out_w, CV_32FC3);

    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            for (int ch = 0; ch < out_c; ch++)
            {
                // Index in NCHW: (n * out_c + ch) * out_h * out_w + i * out_w + j
                // Index in HWC: i * out_w * out_c + j * out_c + ch
                float val = h_output_nchw[(ch * out_h + i) * out_w + j];
                // Clamp values to [0, 1] and convert to Vec3f
                h_output.at<cv::Vec3f>(i, j)[ch] = std::max(0.0f, std::min(1.0f, val));
            }
        }
    }

    // 20. Save output image
    cv::Mat h_output_uint8;
    h_output.convertTo(h_output_uint8, CV_8UC3, 255.0);

    // Generate the output file name based on the input file name.
    std::string output_image_path = genOutputFileName(input_image_path);
    if (output_image_path.empty()) {
        return -1;
    }

    cv::imwrite(output_image_path, h_output_uint8);
    std::cout << "Output image saved to: " << output_image_path << std::endl;

    // 21. Cleanup
    if (d_workspace)
        cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return 0;
}