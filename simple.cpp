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
void quarxz(float* x, float* r, float* z, float* c, const int M, const int N, const int O);

std::vector<float> getdata(std::string filename);
std::vector<int> getint(std::string filename);

cl::Program program;

cl::Context context;

cl::Device device;



int main() {

	const int M = 16;
	const int N = 16;
	const int O = 16;

	const size_t ROWS_A = M;
	const size_t COLS_A = O;
	const size_t COLS_B = N;
	clock_t start, end;

	std::vector<float> a;
	
	a=getdata("id.txt");

	std::vector<int> a1;
	a1 = getint("fix.txt");

	std::vector<float> b;
	b= getdata("x.txt");
	//b = a;

	std::vector<float> z;
	z = getdata("z.txt");
	//std::cout << b[1] <<std::endl;

	//std::vector<float> cp(ROWS_A * COLS_A);
	std::vector<float> cp(27 * 27);
	
	//std::cout << "A matrix size:" << a.size() << std::endl;

	// construct the desing matrix
	
	const int A = a1.size();

	std::set<int> t;
	for (int i = 0; i < A; i++) {
		t.insert(a1[i]);
	}

	std::vector<int> get(t.begin(), t.end());
	const int lengthuni = get.size();

	
	/*for (int j = 0; j<lengthuni; j++) {
		std::cout<< get[j]<<std::endl;
	}*/

	//std::cout << lengthuni << std::endl;

	std::vector<float> c(A * lengthuni);

	initializeDevice();

	start = clock();

	strucmatrix(a1.data(), get.data(), c.data(), A, lengthuni);

	//if (c == b) { std::cout << "same strucx" << std::endl; }

	/*for (int i = 0; i < 32; i++) {

		//std::cout << a1[i] << std::endl;

		std::cout << "a: " << a[i] << "\t";
	}*/

	
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
	//transpose(b.data(), cp.data(), N, 8);
	//parmamu(a.data(), b.data(), cp.data(), M, N, O);
	//quar(b.data(), a.data(), cp.data(), 8, N );
	// quarxz(c.data(), a.data(), z.data(), cp.data(), 8, N, 19);
	//rbind(a.data(), b.data(), cp.data(), M, N, O);
	//pretreat(a.data(),cp.data(), M, 0);
	CMatrix(c.data(), z.data(), a.data(), a.data(), cp.data(), 8, 19, 16);

	end = clock();
	double parTime = ((double)10e2 * (end - start)) / CLOCKS_PER_SEC;

	std::cout << " Total execution time is: " << parTime << " ms." << std::endl;

	//write the vector into a outputfile
	

	std::ofstream outfile("output.txt");
	std::ostream_iterator<int> outiterator(outfile, "\n");

	for (const auto& t : cp) {
		outfile << t << std::endl;
	};
	
	
	for (int i = 0; i < cp.size(); i++) {

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
	const int T = xM + zM;
/*Notice x matrix from the function strucmatrix is a transposed result the function quar based on this transposed result*/
	//quarter matrix
	std::vector<float>  q1(xM*xM), q2(zM*xM), q3(zM * xM), q4(zM * zM);

	//half matrix
	std::vector<float> h1(xM * T), h2(zM * T);

	//get quarter matrix
	quar(x, r, q1.data(), xM, N);
	quar(z, r, q4.data(), zM, N);
	quarxz(x, r,z, q2.data(), xM, N,zM);
	quarxz(z, r, x,q3.data(), zM, N,xM);

	//get half matrix
	cbind(q1.data(), q2.data(),h1.data(),xM,zM,xM );
	cbind(q3.data(), q4.data(), h2.data(), xM, zM, zM);

	//get c matrix

	rbind(h1.data(), h2.data(),c, xM, zM, T);


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

	/**
	* Set kernel arguments.
	**/

	cl::Event eventname1, eventname2, eventname3, eventname4;
	std::vector<cl::Event> events;


	cl::Kernel kernel0(program, "transpose");

	kernel0.setArg(0, xBuf);
	kernel0.setArg(1, tempBuftx);
	kernel0.setArg(2, sizeof(unsigned int), &N);
	kernel0.setArg(3, sizeof(unsigned int), &M);

	cl::Kernel kernel1(program, "mamu");

	kernel1.setArg(0, xBuf);
	kernel1.setArg(1, rBuf);
	kernel1.setArg(2, tempBufxr);
	kernel1.setArg(3, sizeof(unsigned int), &M);
	kernel1.setArg(4, sizeof(unsigned int), &N);
	kernel1.setArg(5, sizeof(unsigned int), &N);

	cl::Kernel kernel2(program, "mamu");

	kernel2.setArg(0, tempBufxr);
	kernel2.setArg(1, tempBuftx);
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
	
	queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(N, M), cl::NullRange, &events, &eventname2);
	events.push_back(eventname2);
	eventname2.waitForEvents(events);

	queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(M, N), cl::NullRange, &events, &eventname3);
	events.push_back(eventname3);
	eventname3.waitForEvents(events);	

	
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * M * sizeof(float), c,&events, &eventname4);
	events.push_back(eventname4);
	eventname4.waitForEvents(events);
	queue.finish();
}

void quarxz(float* x, float* r, float* z, float* c, const int M, const int N,const int O) {

	/**
	* Create buffers and allocate memory on the device.
	**/
	cl::Buffer xBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), x);
	cl::Buffer rBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), r);
	cl::Buffer zBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * O * sizeof(float), z);
	cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, M * O * sizeof(float));

	cl::Buffer tempBuftz(context, CL_MEM_READ_WRITE, O * N * sizeof(float));
	cl::Buffer tempBufxr(context, CL_MEM_READ_WRITE, M * N * sizeof(float));


	/**
	* Set kernel arguments.
	**/

	cl::Event eventname1, eventname2, eventname3, eventname4;
	std::vector<cl::Event> events;


	cl::Kernel kernel0(program, "transpose");

	kernel0.setArg(0, zBuf);
	kernel0.setArg(1, tempBuftz);
	kernel0.setArg(2, sizeof(unsigned int), &N);
	kernel0.setArg(3, sizeof(unsigned int), &O);

	cl::Kernel kernel1(program, "mamu");

	kernel1.setArg(0, xBuf);
	kernel1.setArg(1, rBuf);
	kernel1.setArg(2, tempBufxr);
	kernel1.setArg(3, sizeof(unsigned int), &M);
	kernel1.setArg(4, sizeof(unsigned int), &N);
	kernel1.setArg(5, sizeof(unsigned int), &N);

	cl::Kernel kernel2(program, "mamu");

	kernel2.setArg(0, tempBufxr);
	kernel2.setArg(1, tempBuftz);
	kernel2.setArg(2, cBuf);
	kernel2.setArg(3, sizeof(unsigned int), &M);
	kernel2.setArg(4, sizeof(unsigned int), &O);
	kernel2.setArg(5, sizeof(unsigned int), &N);


	/**
	* Execute the kernel function and collect its result.
	**/

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	queue.enqueueNDRangeKernel(kernel0, cl::NullRange, cl::NDRange(O, N), cl::NullRange, 0, &eventname1);
	events.push_back(eventname1);
	

	queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(N, M), cl::NullRange, &events, &eventname2);
	events.push_back(eventname2);
	eventname2.waitForEvents(events);

	//queue.enqueueReadBuffer(tempBufxr, CL_TRUE, 0, N * M * sizeof(float), c);

	queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(O,N), cl::NullRange, &events, &eventname3);
	events.push_back(eventname3);
	eventname3.waitForEvents(events);
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * O * sizeof(float), c, &events, &eventname4);
	events.push_back(eventname4);
	eventname4.waitForEvents(events);
	queue.finish();
}