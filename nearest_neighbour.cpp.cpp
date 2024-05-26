#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mpi.h>


// function to read in a list of 3D coordinates from an .xyz file
// input: the name of the file
std::vector < std::vector < double > > read_xyz_file(std::string filename, int& N, double& L){

  // open the file
  std::ifstream xyz_file(filename);

  // read in the number of atoms
  xyz_file >> N;
  
  // read in the cell dimension
  xyz_file >> L;
  
  // now read in the positions, ignoring the atomic species
  std::vector < std::vector < double > > positions;
  std::vector < double> pos = {0, 0, 0};
  std::string dummy; 
  for (int i=0;i<N;i++){
    xyz_file >> dummy >> pos[0] >> pos[1] >> pos[2];
    positions.push_back(pos);           
  }
  
  // close the file
  xyz_file.close();
  
  return positions;
  
}


//accepts row index, column index and column size
//returns 1d index of a matrix
int flat_index(int i, int j, int ncol){
   return i * ncol + j;
}


//accepts a matrix
//returns flattened matrix as 1d array
std::vector<double> flatten_matrix(std::vector<std::vector<double>> M){
   int nrow = M.size();
   int ncol = M[0].size();
   std::vector<double> m;
   for (int i = 0; i < nrow; i++){
      for (int j = 0; j < ncol; j++){
         m.push_back(M[i][j]);
      }
   }
   return m;
}


//accepts a 1d array
//returns the average values of the entries in the 1d array
double average_neighbour(std::vector<int> N){
    double sum = 0.0;
    for (int i = 0 ; i < N.size(); i++){
        sum += N[i];
    }
    double average = sum / N.size();
    return average;
}


//accepts flattened matrix containing particles coordinates, 1d array to store the neighbouring particles count, length of neighbour radius
//accepts thread ID, number of total threads and MPI method (0 default for 1d Block and 1 for 1d cyclic)
void brute_force(std::vector<double> A, std::vector<int>& neighbour, double r, double& time, int iproc, int nproc, int method = 0){
    //define the number of particle by the row size of the actual matrix
    int nrow = A.size() / 3;  
    int ncol = 3;
    //starts the timing for the computation of nearest neighbour
    double start = MPI_Wtime();
    //compute brute force approach using the root thread if the job run by one thread 
    if (nproc == 1){ 
        for (int i = 0; i < nrow; i++){
            for (int j = 0; j < nrow; j++){
                double d = 0;
                //for all particle index by i and j where i is not j do
                if (i != j){
                    //calculate the euclidean distance between them
                    for (int k = 0; k < 3; k++){
                    //need to convert the 3d index of i and j into 1d array index
                    int i_flat = flat_index(i, k, ncol);
                    int j_flat = flat_index(j, k, ncol);
                    d += (A[i_flat] - A[j_flat]) * (A[i_flat] - A[j_flat]);
                    }
                    //if the distance between particles larger than input radius enumerate count particle j as neighbour of particle i
                    if (d < r * r){
                        neighbour[i]++;
                    }
                }   
            }
        }
    }
    //use MPI routine if job runs by more than one thread
    else{
        //if method 0 which is 1d Block do
        if (method == 0){
            //initialise empty array to store the neighbour count for all particles
            std::vector<int> N(nrow, 0);
            //threads other than root thread which is 0 are used to perform nearest neighbour count
            if(iproc != 0){
                //initialise the size of blocks
                int dk = nrow / (nproc - 1);
                //initialise the 3d index of first particle in the blocks 
                int k0 = (iproc - 1) * dk;
                //initialise the 3d index of last particle in the blocks
                int k1 = (iproc) * dk;
                //change the index of the last particle in the blocks to the index of the last particle if it is too large
                if (iproc == nproc - 1){
                    k1 = nrow;
                }
                //perform brute force approach for all particles in the preset block
                for (int i = k0; i < k1; i++){
                    for (int j = 0; j < nrow; j++){
                        double d = 0;
                        if (i != j){
                            for (int k = 0; k < 3; k++){
                                int i_flat = flat_index(i, k, ncol);
                                int j_flat = flat_index(j, k, ncol);
                                d += (A[i_flat] - A[j_flat]) * (A[i_flat] - A[j_flat]);
                            }
                            if (d < r * r){
                                N[i]++;
                            }
                        }
                    }
                }   
            }
            //call MPI_Reduce to add the neighbour count by each threads onto the main array containing all neighbour counts in the root thread
            MPI_Reduce(N.data(), neighbour.data(), nrow, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
        }
        //if method 1 which is 1d cyclic 
        if (method == 1){
            //initialise empty array to store the neighbour count for all particles
            std::vector<int> N(nrow, 0);
            //threads other than root thread which is 0 are used to perform nearest neighbour count
            if(iproc != 0){
                //define the maximum number of possible cycles that can be made by working threads
                int dk = nrow / (nproc - 1) + 1;
                //for each number of cycle
                for (int i = 0; i < dk; i++){
                    //initialise index of particles in the cycle of the chosen thread
                    int new_index = (iproc - 1) + (nproc - 1) * i;
                    //if the particle index is less than the total number of particles do
                    if (new_index <= nrow){
                        //perform brute force approach for all particles in the cycle of the chosen thread
                        for (int j = 0; j < nrow; j++){
                            double d = 0;
                            if (new_index != j){
                                for (int k = 0; k < 3; k++){
                                    int i_flat = flat_index(new_index, k, ncol);
                                    int j_flat = flat_index(j, k, ncol);
                                    d += (A[i_flat] - A[j_flat]) * (A[i_flat] - A[j_flat]);
                                }
                                if (d < r * r){
                                    N[new_index]++;
                                }
                            }
                        }
                    }
                }   
            }
            //call MPI_Reduce to add the neighbour count by each threads onto the main array containing all neighbour counts in the root thread
            MPI_Reduce(N.data(), neighbour.data(), nrow, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
        }
        //if input for method is other than 1 and 0 print message to let the user knows
        if (abs(method) > 1 & iproc == 0){
            std::cout << std::endl;
            std::cout << "ERROR: For argument method only accept 0 to implement 1d Block or 1 to implement 1d Cyclic." <<std::endl;
            std::cout << std::endl;
        } 
    }
    //obtain the runtime
    double finish = MPI_Wtime();
    double t = finish - start;
    //call MPI_Reduce to submit the runtime of all working threads to the root threads
    //and choose the maximum runtime from all using MPI_MAX
    MPI_Reduce(&t, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}


//print the results from the nearest neighbour algorithm
//accepts 1d array containing neighbouring particles count, the total time, thread ID and number of total threads
void write_summary(std::vector<int> neighbour, double total_time, int iproc, int nproc){
    //only the root thread will print the information 
    if (iproc == 0) {
        //number of total threads
        std::cout << "Number of process : " << nproc << std::endl;
        //number of total particles
        std::cout << "Total particles : " << neighbour.size() << std::endl;
        //total runtime
        std::cout << "Running time : " << total_time << std::endl;
        //average, maximum and minimum count of neighbours
        std::cout << "Average neighbour : " << average_neighbour(neighbour) << std::endl;
        std::cout << "Maximum neighbour : " << *std::max_element(neighbour.begin(), neighbour.end()) << std::endl;
        std::cout << "Minimum neighbour : " << *std::min_element(neighbour.begin(), neighbour.end()) << std::endl;
        std::cout << std::endl;
    }
}


//write the entire neighbouring particle count into a text file
//accepts name of file, 1d array of neighbouring particles count and thread ID
void write_file(std::string filename, std::vector<int>& data, int iproc){
    //only the root thread will write the file
    if (iproc == 0){
        std::ofstream outfile;
        int nrow = data.size();
        outfile.open(filename);
        outfile << "Particle Id," << "Number of neighbours" << std::endl; 
        for (int i = 0; i < nrow; i++){
            outfile << i << "," << data[i] << std::endl;
        }
        outfile.close();
        std::cout << "Finish writing data into " << filename << " file." << std::endl;
    }
}


int main(int argc, char** argv){
    //MPI routine statements
    MPI_Init(&argc, &argv);
    int nproc, iproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    //initialise the number of particles in all systems
    int N1 = 120;
    int N2 = 10549;
    int N3 = 147023;

    //initialise the length of box in all systems
    double L1 = 18.0;
    double L2 = 100.0;
    double L3 = 200.0;

    //initialise the radius of the neighbour for all systems
    double r_c = 9.0;

    //initialise empty 1d array to store flattened matrices containing coordinates of particles in all systems
    std::vector<double> ar_120;
    std::vector<double> ar_10549;
    std::vector<double> ar_147023;

    //initialise empty 1d array to store the neighbouring particle counts
    std::vector<int> neighbour1(N1, 0);
    std::vector<int> neighbour2(N2, 0);
    std::vector<int> neighbour3(N3, 0);

    //initialise the instances to record the runtime of the nearest neighbour algorithm
    double t1, t2, t3; 

    //request the root thread to read and convert the matrices in all files into 1d arrays
    if (iproc == 0){
        ar_120 = flatten_matrix(read_xyz_file("argon120.xyz", N1, L1));
        ar_10549 = flatten_matrix(read_xyz_file("argon10549.xyz", N2, L2));
        ar_147023 = flatten_matrix(read_xyz_file("argon147023.xyz", N3, L3));   
    }
    
    //for all other working threads 
    if (nproc > 1){
        ar_120.resize(N1 * 3);
        ar_10549.resize(N2 * 3);
        ar_147023.resize(N3 * 3);

        //request a broadcasting of 1d array of particle coordinates from the root thread
        MPI_Bcast(ar_120.data(), N1 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ar_10549.data(), N2 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ar_147023.data(), N3 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
 
    //run the nearest neighbour algorithm 
    //in this example the algorithm is run using 1d Cyclic approach
    brute_force(ar_120, neighbour1, r_c, t1, iproc, nproc, 1); 
    brute_force(ar_10549, neighbour2, r_c, t2, iproc, nproc, 1); 
    brute_force(ar_147023, neighbour3, r_c, t3, iproc, nproc, 1); 
    
    //uncomment the below lines and comment the above lines to run using 1d Block approach
    //brute_force(ar_120, neighbour1, r_c, t1, iproc, nproc); 
    //brute_force(ar_10549, neighbour2, r_c, t2, iproc, nproc); 
    //brute_force(ar_147023, neighbour3, r_c, t3, iproc, nproc); 

    //print out the results of the computations
    write_summary(neighbour1, t1, iproc, nproc);  
    write_summary(neighbour2, t2, iproc, nproc); 
    write_summary(neighbour3, t3, iproc, nproc); 

    //write the neighbouring particles count into csv files
    write_file("neighbour1.csv", neighbour1, iproc);
    write_file("neighbour2.csv", neighbour2, iproc);
    write_file("neighbour3.csv", neighbour3, iproc);

    MPI_Finalize();
    return 0;
}