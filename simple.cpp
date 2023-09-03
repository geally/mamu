

#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <CL/opencl.hpp>

//To do: add the calculation time counting 
// compaire the difference of different mapping

cl::Device getDefaultDevice();

void initializeDevice();
void parmamu(int* a, int* b, int* c, const int M,const int N, const int O);
void inverse(float * in,float * out, const int M, const int N);
void anothertest(int* a, int* b, int* c, const int M, const int N, const int O);
std::vector<float> getdata(std::string filename);

cl::Program program;
cl::Context context;
cl::Device device;

int main() {

	const int M =  3;
	const int N = 6;
	const int O =  3;

	const size_t ROWS_A = M;
	const size_t COLS_A = O;

	const size_t COLS_B = N;

	std::vector<float> a;
	a = getdata("testdata.txt");
	std::vector<float> b;
	b = { 1,2,3,4,5,6 };
	std::vector<float> cp(ROWS_A * COLS_B);

	initializeDevice();

	

	//parmamu(a.data(), b.data(), cp.data(), M, N, O);

	inverse(a.data(), cp.data(), M, N);

	//anothertest(a.data(), b.data(), cp.data(), M, N, O);
	//write the vector into a outputfile
	std::ofstream outfile("output.txt");
	//std::ostream_iterator<int> outiterator(outfile, "\n");
	for (const auto &t : cp) {
		outfile << t << std::endl;
	};

	std::cout << "The first 10 element of results are:";
	for (int i = 0; i < 10; i++) {


		std::cout << cp[i] << "\t";
	}
	
	return 0;
}

cl::Device getDefaultDevice(){
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
		std::cerr	 << "NO PLATFORMS" << std::endl;
		exit(1);
	}

	auto platform = platforms.front();
	std::vector<cl::Device> devices;



	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	

	if (devices.empty()) {
		std::cerr << "No devices" << std::endl;
		exit(1);
	}

	return devices.front();
}

void initializeDevice() {
	device = getDefaultDevice();
	std::cout << "Max compute units " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	


	std::ifstream kernel_file("simplemm.cl");

	std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

	//std::cout << src << std::endl;
	

	/*std::string kernel = "__kernel void mamu(__global int* a,"
		"__global int* b,"
		"__global int* c,"
		"const int M,"
		"const int N,"
		"const int K){"
		"int colIndex = get_global_id(0);"
		"int rowIndex = get_global_id(1);"
		"int index = (rowIndex * N) + colIndex;"
		"int sum = 0;"
		"for (int k = 0; k < K; k++) {sum += a[rowIndex * K + k] * b[k * N + colIndex];}"
		" c[index] = sum;}";*/

	/**
	 * Compile kernel program which will run on the device.
	 * */

	//std::cout << kernel << std::endl;

	cl::Program::Sources sources;
	
	sources.push_back({ src.c_str(),src.length() + 1 });

	context = cl::Context(device);
	program = cl::Program(context,sources);

	auto err = program.build();
	if (err != CL_BUILD_SUCCESS) {
		std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
			<< "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
}

void parmamu(int* a, int* b, int* c,  const int M, const int N,const int O) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O* M * sizeof(int), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O* N * sizeof(int), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "mamu");

	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, sizeof(unsigned int), &M);
	kernel.setArg(4, sizeof(unsigned int), &N);
	kernel.setArg(5, sizeof(unsigned int), &O);

	/**
	* Execute the kernel function and collect its result.
	**/
	
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N,M));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
	queue.finish();
}


void inverse(float* in, float* out, const int M, const int N) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), in);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * N * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "inverse");

	kernel.setArg(0, inBuf);
	kernel.setArg(1, outBuf);
	kernel.setArg(2, sizeof(unsigned int), &M);
	kernel.setArg(3, sizeof(unsigned int), &N);


	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, M * N * sizeof(int), out);
	queue.finish();
}

void anothertest(int* a, int* b, int* c, const int M, const int N, const int O) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * M * sizeof(int), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * N * sizeof(int), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "anothertest");

	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, sizeof(unsigned int), &M);
	kernel.setArg(4, sizeof(unsigned int), &N);
	kernel.setArg(5, sizeof(unsigned int), &O);

	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, M));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
	queue.finish();
}

std::vector<float> getdata(std::string filename) {
	 std::ifstream infile(filename);
	std::vector<float> vec;
	int line;
	while (infile >> line ) {
		vec.push_back(line);
	};
	return vec;
};