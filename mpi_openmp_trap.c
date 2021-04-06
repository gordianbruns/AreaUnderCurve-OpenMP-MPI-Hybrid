/* File:    mpi_openmp_trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n, t
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: mpigcc -g -Wall -fopenmp -o mpi_openmp_trap mpi_openmp_trap.c
 * Usage:   mpirun --mca btl_tcp_if_include 192.168.5.0/24 -np <int > 1> -hostfile whedon-hosts --map-by node mpi_openmp_tra -a <double> -b <double > a> -n <int > 0> -t <int > 0>p
 *
 * Note:    The function f(x) is hardwired.
 *
 * IPP:     Section 3.2.1 (pp. 94 and ff.) and 5.2 (p. 216)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

int my_node_rank;
int comm_sz;
MPI_Comm comm;


double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, double h, int threads_requested);
void Get_arg(int argc, char* argv[], int* debug, double* a, double* b, long* n, int* threads);

int main(int argc, char* argv[]) {
   double  integral;   /* Store result in integral   */
   double  a, b;       /* Left and right endpoints   */
   long     n;          /* Number of trapezoids       */
   double  h;          /* Height of trapezoids       */
   double node_integral;
   int threads_requested = 1, provided;
   int i, nlength;
   int debug = 0;
   struct timeval startTime, stopTime;
   double wallTime;
   char name[80];

   comm = MPI_COMM_WORLD;

   MPI_Status status;

   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
   MPI_Comm_rank(comm, &my_node_rank);
   MPI_Comm_size(comm, &comm_sz);

   Get_arg(argc, argv, &debug, &a, &b, &n, &threads_requested);

   if (debug) {
	MPI_Get_processor_name(name, &nlength);
	name[nlength] = '\0';
	fprintf(stderr, "my_node = %s, my_node_rank = %d, comm_sz = %d, threads_requested = %d\n", name, my_node_rank, comm_sz, threads_requested);
   }

   if (my_node_rank == 0) {
	int n_workers = comm_sz - 1;
	double step;
	int remainder, node_n;
	double copy_a = a;
	double copy_b = b;

	if (debug) {
	    switch(provided) {
		case MPI_THREAD_SINGLE:
		    fprintf(stderr, "The MPI binding is providing MPI_THREAD_SINGLE\n");
		    break;
		case MPI_THREAD_FUNNELED:
		    fprintf(stderr, "The MPI binding is providing MPI_THREAD_FUNNELED\n");
		    break;
		case MPI_THREAD_SERIALIZED:
		    fprintf(stderr, "The MPI binding is providing MPI_THREAD_SERIALIZED\n");
		    break;
		case MPI_THREAD_MULTIPLE:
		    fprintf(stderr, "The MPI binding is providing MPI_THREAD_MULTIPLE\n");
		    break;
		default:
		    fprintf(stderr, "Unknown MPI thread level: %d\n", provided);
		    break;
	    }
	}

	if (threads_requested < 1) {
	    printf("Use -d for debugging\n");
	    fprintf(stderr, "usage: mpirun -np <int > 1> -hostfile whedon-hosts --map-by node mpi_openmp_trap -a <double> -b <double and > a> -n <int > 0>\n");
            exit(-1);
	}

	if (b < a || n <= 0) {
	    printf("Use -d for debugging\n");
	    fprintf(stderr, "usage: mpirun -np <int > 1> -hostfile whedon-hosts --map-by node mpi_openmp_trap -a <double> -b <double and > a> -n <int > 0>\n");
	    exit(-1);
	}

	if (n < n_workers) {
	    n_workers = n;
	}

	step = (b - a) / n_workers;
	remainder = n % n_workers;
	
	if (gettimeofday(&startTime, NULL) != 0) {
	    perror("gettimeofday() startTime failed");
	    exit(-1);
	}

	int original;

	for (i = 1; i <= n_workers; i++) {
	    node_n = n / n_workers;
	    b = a + step;
	    original = node_n;
	    if (remainder > 0) {
		b += (copy_b - copy_a) / n;
		node_n += 1;
		remainder -= 1;
	    }
	    MPI_Send(&a, 1, MPI_DOUBLE, i, 1, comm);
	    MPI_Send(&b, 1, MPI_DOUBLE, i, 1, comm);
	    MPI_Send(&node_n, 1, MPI_INT, i, 1, comm);

	    if (original != node_n) {
		a += (copy_b - copy_a) / n;
	    }

	    a += step;
	}

	integral = 0;

	for (i = 1; i <= n_workers; i++) {
	    MPI_Recv(&node_integral, 1, MPI_DOUBLE, i, 2, comm, &status);
	    integral += node_integral;
	}

	if (gettimeofday(&stopTime, NULL) != 0) {
	    perror("gettimeofday() stopTime failed");
	    exit(-1);
	}

	wallTime = (double)(stopTime.tv_sec - startTime.tv_sec) + (double)((stopTime.tv_usec - startTime.tv_usec) * (double)0.000001);

	printf("With n = %li trapezoids, our estimate\n", n);
	printf("of the integral from %f to %f = %.15f\n", copy_a, copy_b, integral);
	printf("Elapsed time = %.6fs\n", wallTime);


   } else {

	double node_a, node_b;
	int node_n;
	double my_result;

	MPI_Recv(&node_a, 1, MPI_DOUBLE, 0, 1, comm, &status);
	MPI_Recv(&node_b, 1, MPI_DOUBLE, 0, 1, comm, &status);
	MPI_Recv(&node_n, 1, MPI_INT, 0, 1, comm, &status);

	h = (node_b - node_a) / node_n;
   	my_result = Trap(node_a, node_b, node_n, h, threads_requested);

	MPI_Send(&my_result, 1, MPI_DOUBLE, 0, 2, comm);
   }
   MPI_Finalize();
   return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, int n, double h, int threads_requested) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   # pragma omp parallel for num_threads(threads_requested) reduction(+: integral) schedule(static, 1)
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  /* Trap */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
   double return_val;

   return_val = x*x;
   return return_val;
}  /* f */

void Get_arg(int argc, char* argv[], int* debug, double* a, double* b, long* n, int* threads) {
   int opt = 0;
   char *ptr, err_msg[128] = "usage: mpirun -np <int > 1> -hostfile whedon-hosts --map-by node mpi_openmp_trap -a <double> -b <double and > a> -n <int > 0>\n";

   if (my_node_rank == 0) {
	if (argc < 2) {
	    MPI_Finalize();
	    fprintf(stderr, err_msg);
	    exit(-1);
	}

	if (comm_sz < 2) {
	    MPI_Finalize();
	    fprintf(stderr, err_msg);
	    exit(-1);
	} else {
	    while ((opt = getopt(argc, argv, "d:a:b:n:t:")) != -1) {
		switch(opt) {
		    case 'd':
			*debug = 1;
			break;
		    case 'a':
			*a = strtod(optarg, &ptr);
			break;
		    case 'b':
			*b = strtod(optarg, &ptr);
			break;
		    case 'n':
			*n = atol(optarg);
			break;
		    case 't':
			*threads = strtol(optarg, (char**) NULL, 10);
			break;
		}
	    }
	}
   }
}
