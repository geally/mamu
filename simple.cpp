#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <time.h>
#include <CL/opencl.hpp>


//To do: add the calculation time counting 
// compaire the difference of different mapping

cl::Device getDefaultDevice();

void initializeDevice();
void parmamu(float* a, float* b, float* c, const int M,const int N, const int O);
void inverse2(float* in, float* out, const int M, const int N);
void strucmatrix(int* y, int* uniquey, float* out, const int N, const int lengthuni);
void cbind(float* a, float* b, float* c, const int M, const int N, const int O);
void rbind(float* a, float* b, float* c, const int M, const int N, const int O);
void transpose(float* a, float* b, const int M, const int N);
void CMatrix(float* x, float* z, float* r,float* g, float* c, const int xM, const int zM, const int N);
void quar(float* x, float* r, float* c, const int M, const int N);

std::vector<float> getdata(std::string filename);
std::vector<int> getint(std::string filename);

cl::Program program;

cl::Context context;

cl::Device device;



int main() {

	const int M = 60;
	const int N = 60;
	const int O = 60;

	const size_t ROWS_A = M;
	const size_t COLS_A = O;
	const size_t COLS_B = N;
	clock_t start, end;

	std::vector<float> a;
	
	a=getdata("id.txt");

	std::vector<int> a1;
	a1 = getint("fix.txt");

	//std::vector<float> b;
	//b = a;

	//std::cout << b[1] <<std::endl;

	std::vector<float> cp(ROWS_A * COLS_A);
	
	//std::cout << "A matrix size:" << a.size() << std::endl;

	// construct the desing matrix
	
	const int A = a1.size();

	std::set<int> t;
	for (int i = 0; i < A; i++) {
		t.insert(a1[i]);
	}

	std::vector<int> get(t.begin(), t.end());
	const int lengthuni = get.size();

	for (int j =0; j<lengthuni;j++){
		std::cout<< get[j]<<std::endl;
	}

	//std::cout << lengthuni << std::endl;

	std::vector<float> c(A * lengthuni);

	



	initializeDevice();

	start = clock();

	strucmatrix(a1.data(), get.data(), c.data(), A, lengthuni);
	//inverse(a.data(), cp.data(), M, N);

	//compmamul(a.data(), b.data(), cp.data(), M,N);

	
	
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
		//end = clock();
		// double parTime = ((double)10e3 * (end - start)) / CLOCKS_PER_SEC;

		// std::cout << "At round"<<i<< "execution time is: " << parTime << " ms." << std::endl;
	}
	*/
	//transpose(a.data(), cp.data(), M, N);
	//parmamu(a.data(), b.data(), cp.data(), M, N, O);

	for (int i = 0; i < 9; i++) {

		//std::cout << a1[i] << std::endl;

		std::cout << "Mc: " << c[i] << "\t";
	}

	quar(a.data(), c.data(), cp.data(), M, N);
	//rbind(a.data(), b.data(), cp.data(), M, N, O);
	//for (int i = 0; i < N; i++)inverse3(a.data(), cp.data(), M, i);
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
	
	for (int i = 0; i < 9; i++) {

		//std::cout << a1[i] << std::endl;

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
	
	std::ifstream kernel_file("simplemm.cl");

	std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
	

	/**
	 * Compile kernel program which will run on the device.
	 * */

	//std::cout << kernel << std::endl;

	cl::Program::Sources sources;
	
	sources.push_back({ src.c_str(),src.length() + 1 });

	context = cl::Context(device);
	program = cl::Program(context,sources);
	//const char* option ="-cl-std=CL2.0 -D CL_VERSION_2_0";
	//program.compile(option);


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


std::vector<float> getdata(std::string filename) {
	 std::ifstream infile(filename);
	std::vector<float> vec;

	float line;
	while (infile >> line ) {
		vec.push_back(line);
	};
	return vec;
};

std::vector<int> getint(std::string filename) {
	std::ifstream infile(filename);
	std::vector<int> vec;

	int line;
	while (infile >> line) {
		vec.push_back(line);
	};
	return vec;
};

void strucmatrix(int* y,int *uniquey,float * out, const int N,const int lengthuni){
	
	cl::Buffer inBuf1(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR, N  * sizeof(int), y);
	cl::Buffer inBuf2(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR, lengthuni * sizeof(int), uniquey);
	cl::Buffer outBuf(context, CL_MEM_READ_WRITE, N * lengthuni * sizeof(int));
	cl::Buffer out2Buf(context, CL_MEM_READ_WRITE, N * lengthuni * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel1(program, "refmatrix");

	kernel1.setArg(0, inBuf1);
	kernel1.setArg(1, inBuf2);
	kernel1.setArg(2, outBuf);
	kernel1.setArg(3, sizeof(unsigned int),&N);

	cl::Kernel kernel2(program, "constructmatrix");
	kernel2.setArg(0, inBuf1);
	kernel2.setArg(1, inBuf2);
	kernel2.setArg(2, out2Buf);
	kernel2.setArg(3, outBuf);
	kernel2.setArg(4, sizeof(unsigned int), &N);

	
	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	cl::Event eventname1;
	cl::Event eventname2;
	std::vector<cl::Event> events;

	queue.enqueueNDRangeKernel( kernel1, cl::NullRange, cl::NDRange(N , lengthuni), cl::NullRange,  NULL,&eventname1);
	eventname1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	events.push_back(eventname1);
	queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(N, lengthuni), cl::NullRange, &events, &eventname2);
	events.push_back(eventname2);
	queue.enqueueReadBuffer(out2Buf, CL_TRUE, 0, N * lengthuni * sizeof(float), out,&events);
	queue.finish();
};

void cbind(float* a, float* b, float* c, const int M, const int N, const int O) {
	const int T = M +N;
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * M * sizeof(float), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * N * sizeof(float), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, T * O * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "cbind");

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

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(T, O));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, T * O * sizeof(float), c);
	queue.finish();
}
void rbind(float* a, float* b, float* c, const int M, const int N, const int O) {
	const int T = M + N;
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * M * sizeof(float), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, O * N * sizeof(float), b);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, T * O * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "rbind");

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

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange( O,T));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, T * O * sizeof(float), c);
	queue.finish();
}

void transpose(float* a, float* b, const int M, const int N) {
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), a);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * N * sizeof(float));
	/**
	* Set kernel arguments.
	**/
	cl::Kernel kernel(program, "transpose");

	kernel.setArg(0, aBuf);
	kernel.setArg(1, cBuf);
	kernel.setArg(2, sizeof(unsigned int), &M);
	kernel.setArg(3, sizeof(unsigned int), &N);

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M, N));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(float), b);
	queue.finish();
}

void CMatrix(float* x, float* z, float* r, float* g, float* c, const int xM, const int zM, const int N) {
	//const int T = xM + zM;
	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer xBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N* xM * sizeof(float), x);
	//cl::Buffer zBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, zM* N * sizeof(float), z);
	cl::Buffer rBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N* N * sizeof(float), r);
	//cl::Buffer gBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, zM* zM * sizeof(float), g);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, xM * xM * sizeof(float));

	cl::Buffer tempBufxx(context, CL_MEM_READ_WRITE, xM* xM * sizeof(float));
	//cl::Buffer tempBufxz(context, CL_MEM_READ_WRITE, xM* zM * sizeof(float));
	//cl::Buffer tempBufzx(context, CL_MEM_READ_WRITE, xM* zM * sizeof(float));
	//cl::Buffer tempBufzz(context, CL_MEM_READ_WRITE, zM* zM * sizeof(float));

	/**
	* Set kernel arguments.
	**/

	cl::Event eventname1, eventname2, eventname3, eventname4;
	std::vector<cl::Event> events;


	cl::Kernel kernel0(program, "trimax");

	kernel0.setArg(0, xBuf);
	kernel0.setArg(1, rBuf);
	kernel0.setArg(2, cBuf);
	kernel0.setArg(3, sizeof(unsigned int), &xM);
	kernel0.setArg(4, sizeof(unsigned int), &N);



	/*cl::Kernel kernel1(program, "mamu");

	kernel1.setArg(0, xBuf);
	kernel1.setArg(1, zBuf);
	kernel1.setArg(2, cBuf);
	kernel1.setArg(3, sizeof(unsigned int), &xM);
	kernel1.setArg(4, sizeof(unsigned int), &N);
	kernel1.setArg(5, sizeof(unsigned int), &xM);

	cl::Kernel kernel2(program, "mamu");

	kernel2.setArg(0, zBuf);
	kernel2.setArg(1, xBuf);
	kernel2.setArg(2, cBuf);
	kernel2.setArg(3, sizeof(unsigned int), &xM);
	kernel2.setArg(4, sizeof(unsigned int), &N);
	kernel2.setArg(5, sizeof(unsigned int), &xM);

	cl::Kernel kernel3(program, "mamu");

	kernel3.setArg(0,zBuf);
	kernel3.setArg(1, zBuf);
	kernel3.setArg(2, cBuf);
	kernel3.setArg(3, sizeof(unsigned int), &xM);
	kernel3.setArg(4, sizeof(unsigned int), &N);
	kernel3.setArg(5, sizeof(unsigned int), &xM);
	*/
	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel0, cl::NullRange, cl::NDRange( xM,N));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, N* xM * sizeof(float), c);
	queue.finish();
}

void quar(float* x,  float* r, float* c, const int M, const int N) {

	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer xBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), x);
	cl::Buffer rBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), r);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * M * sizeof(float));

	cl::Buffer tempBuftx(context, CL_MEM_READ_WRITE, M * N * sizeof(float));
	cl::Buffer tempBufxr(context, CL_MEM_READ_WRITE, M* N * sizeof(float));
	//cl::Buffer tempBufzx(context, CL_MEM_READ_WRITE, xM* zM * sizeof(float));
	//cl::Buffer tempBufzz(context, CL_MEM_READ_WRITE, zM* zM * sizeof(float));

	/**
	* Set kernel arguments.
	**/

	cl::Event eventname1, eventname2, eventname3, eventname4;
	std::vector<cl::Event> events;


	cl::Kernel kernel0(program, "transpose");

	kernel0.setArg(0, xBuf);
	kernel0.setArg(1, tempBuftx);
	kernel0.setArg(2, sizeof(unsigned int), &M);
	kernel0.setArg(3, sizeof(unsigned int), &N);

	cl::Kernel kernel1(program, "mamu");

	kernel1.setArg(0, tempBuftx);
	kernel1.setArg(1, rBuf);
	kernel1.setArg(2, tempBufxr);
	kernel1.setArg(3, sizeof(unsigned int), &M);
	kernel1.setArg(4, sizeof(unsigned int), &N);
	kernel1.setArg(5, sizeof(unsigned int), &N);

	cl::Kernel kernel2(program, "mamu");

	kernel2.setArg(0, tempBufxr);
	kernel2.setArg(1, xBuf);
	kernel2.setArg(2, cBuf);
	kernel2.setArg(3, sizeof(unsigned int), &M);
	kernel2.setArg(4, sizeof(unsigned int), &M);
	kernel2.setArg(5, sizeof(unsigned int), &N);
	
	
	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel0, cl::NullRange, cl::NDRange(M, N), cl::NullRange,0,&eventname1);
	events.push_back(eventname1);
	eventname1.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(M, N), cl::NullRange, &events, &eventname2);
	events.push_back(eventname2);
	queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(M, M), cl::NullRange, &events, &eventname3);
	events.push_back(eventname3);
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * M * sizeof(float), c,&events);
	queue.finish();
}