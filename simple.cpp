

#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <CL/opencl.hpp>

//To do: add the calculation time counting 
// compaire the difference of different mapping

cl::Device getDefaultDevice();

void initializeDevice();
void parmamu(float* a, float* b, float* c, const int M,const int N, const int O);
void compmamul(float* a, float* b, float* c, const int M, const int N);
void inverse(float * in,float * out, const int M, const int N);
void inverse2(float* in, float* out, const int M, const int N);
void inverse3(float* in, float* out, const int M, const int N);
void pretreat(float* in, float* out, const int M, const int N);
std::vector<float> getdata(std::string filename);

cl::Program program;
cl::Context context;
cl::Device device;


int main() {

	const int M = 20000;
	const int N = 20000;
	const int O = 20000;

	const size_t ROWS_A = M;
	const size_t COLS_A = O;
	const size_t COLS_B = N;
	clock_t start, end;

	std::vector<float> a;
	
	a=getdata("solve.txt");

	std::vector<float> b;
	b = a;

	//std::cout << b[1] <<std::endl;

	std::vector<float> cp(ROWS_A * COLS_B);
	std::cout <<"A matrix size:" << a.size() << std::endl;
	//std::vector<cl_float> cp(4 * 4);

	initializeDevice();

	start = clock();

	//inverse(a.data(), cp.data(), M, N);

	//compmamul(a.data(), b.data(), cp.data(), M,N);

	//parmamu(a.data(), b.data(), cp.data(), M, N, O);
	
	//inverse2 will use the loop which runing on CPU C++

	/*for (int i = 0; i < M; i++) {
		//start = clock();
		float mid = a[i * M + i];
		for (int j = 0; j < M; j++) {
			if (j != i) {
				a[i * M + j] = a[i * M + j] / mid;
				a[j * M + i] = -a[j * M + i] / mid;
			}
			else {
				a[i * M + i] = 1 / a[i * M + i];
				}
		}
		 inverse2(a.data(), cp.data(), M, i);
		 a=cp;
		// end = clock();
		// double parTime = ((double)10e3 * (end - start)) / CLOCKS_PER_SEC;

		// std::cout << "At round"<<i<< "execution time is: " << parTime << " ms." << std::endl;
	}*/

	//inverse 3 will use the swap to exchange the data
	
	for (int i = 0; i < N; i++)inverse3(a.data(), cp.data(), M, i);
	//pretreat(a.data(),cp.data(), M, 0);
	end = clock();
	double parTime = ((double)10e2 * (end - start)) / CLOCKS_PER_SEC;

	std::cout << " Total execution time is: " << parTime << " ms." << std::endl;

	//write the vector into a outputfile
	

	//std::ofstream outfile("output2.txt");
	//std::ostream_iterator<int> outiterator(outfile, "\n");

	/*for (const auto& t : cp) {
		outfile << t << std::endl;
	};*/
	
	//test the result 
	//std::vector<float> temp(ROWS_A * COLS_B);
	//parmamu(a.data(), b.data(), temp.data(), M, N, O);
	std::cout << "The first 9 element of results are:";
	for (int i = 0; i < 9; i++) {

		std::cout << a[i] << std::endl;

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
	//std::cout << "Max compute units " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	//std::cout << "SVM_capability " << device.getInfo<CL_DEVICE_SVM_CAPABILITIES>() << std::endl;
	//printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", CL_DEVICE_LOCAL_MEM_TYPE == 1 ? "local" : "global");

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

void parmamu(float* a, float* b, float* c,  const int M, const int N,const int O) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O* M * sizeof(float), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O* N * sizeof(float), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * N * sizeof(float));
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
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(float), c);
	queue.finish();
}

void compmamul(float* a, float* b, float* c, const int M,const int N) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, 4 * M * sizeof(float), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, 4 * N * sizeof(float), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 4 * 4 * sizeof(float));
	
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "mmmKernel");

	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, sizeof(unsigned int), &M);
	kernel.setArg(4, sizeof(unsigned int), &N);
	
	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, 4));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, 4 * 4 * sizeof(float), c);
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

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M));
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, M * N * sizeof(float), out);
	queue.finish();
}

void inverse2(float* in, float* out, const int M, const int N) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, M * M * sizeof(float), in);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * M * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "inverse2");

	kernel.setArg(0, inBuf);
	kernel.setArg(1, outBuf);
	kernel.setArg(2, sizeof(unsigned int), &M);
	kernel.setArg(3, sizeof(unsigned int), &N);


	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M,M));
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, M * M * sizeof(float), out);
	queue.finish();
}

void inverse3(float* in, float* out, const int M, const int N) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Event event_pretreat, event_inverse;
	cl::vector<cl_event> waitlist[1];
	

	cl::Buffer in1Buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  M * M * sizeof(float),in);
	cl::Buffer in2Buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY , M * M * sizeof(float));
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * M * sizeof(float));
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	//Kernel 0
	cl::Kernel kernel0(program, "pretreat");
	kernel0.setArg(0, in1Buf);
	kernel0.setArg(1, in2Buf);
	kernel0.setArg(2, sizeof(unsigned int), &M);
	

	//Kernel 1
	cl::Kernel kernel1(program, "inverse2");

	kernel1.setArg(0, in2Buf);
	kernel1.setArg(1, outBuf);
	kernel1.setArg(2, sizeof(unsigned int), &M);
	
	


		kernel0.setArg(3, sizeof(unsigned int), &N);
		kernel1.setArg(3, sizeof(unsigned int), &N);
		queue.enqueueNDRangeKernel(kernel0, cl::NullRange, cl::NDRange(M), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(M, M),cl::NullRange);
		
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, M * M * sizeof(float), out);
	queue.finish();
}

void pretreat(float* in, float * out ,const int M, const int N) {
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer inBuf(context, CL_MEM_READ_WRITE |  CL_MEM_COPY_HOST_PTR, M * M * sizeof(float), in);
	cl::Buffer outBuf(context, CL_MEM_READ_WRITE, M * M * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "pretreat");

	kernel.setArg(0, inBuf);
	kernel.setArg(1, outBuf);
	kernel.setArg(2, sizeof(unsigned int), &M);
	kernel.setArg(3, sizeof(unsigned int), &N);

	cl::Kernel kernel2(program, "pretreat");

	kernel2.setArg(0, outBuf);
	kernel2.setArg(1, inBuf);
	kernel2.setArg(2, sizeof(unsigned int), &M);
	kernel2.setArg(3, sizeof(unsigned int), &N);


	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	for (int i = 0; i < 2; i++) {
		if (i == 0){
			queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(M));
			
		}
		else {
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(M));
		}
		
	}
	//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M));
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, M * M * sizeof(float), out);
	queue.finish();
}

std::vector<float> getdata(std::string filename) {
	 std::ifstream infile(filename);
	std::vector<float> vec;

	float line;
	while (infile >> line ) {
		vec.push_back(line);
	};
	return vec;
};