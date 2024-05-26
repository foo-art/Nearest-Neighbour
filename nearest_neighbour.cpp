#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
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


int flat_index(int i, int j, int ncol){
   return i * ncol + j;
}


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


void print_vector(std::vector<int> N){
    for (int i = 0; i < N.size(); i++){
        std::cout << i << ", " << N[i] << std::endl;
    }
    std::cout << std::endl;   
}


double average_neighbour(std::vector<int> N){
    double sum = 0.0;
    for (int i = 0 ; i < N.size(); i++){
        sum += N[i];
    }
    double average = sum / N.size();
    //std::cout << sum << ", " << N.size() << std::endl;
    return average;
}


void brute_force(std::vector<double> A, double r, std::vector<int>& neighbour, int iproc, int nproc){
    int n = A.size() / 3;
    int ncol = 3;
    
    int dk = n / nproc;
    int k0 = iproc * dk;
    int k1 = (iproc + 1) * dk;
    //std::cout << k1 << std::endl;
    std::vector<int> N(n, 0);

    if (iproc == nproc - 1){
      k1 = A.size() / 3;
    }

    for (int i = k0; i < k1; i++){
      for (int j = 0; j < n; j++){
          double d = 0;
          if (i != j){
              for (int k = 0; k < 3; k++){
                  int i_flat = flat_index(i, k, ncol);
		  int j_flat = flat_index(j, k, ncol);
                  d += (A[i_flat] - A[j_flat]) * (A[i_flat] - A[j_flat]);
		  //std::cout << k << ", " << i_flat << ", " << j_flat << std::endl;
              }
              if (d < r * r){
                  N[i]++;
             }
          }
       }
    }
    if (nproc == 1){
      for (int i = 0; i < n; i++){
         neighbour[i] = N[i];
         //std::cout << N[i] << std::endl;
      }
    }
    if (nproc > 1){
      MPI_Reduce(N.data(), neighbour.data(), n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);    
    }
}


void write_summary(std::vector<int> neighbour, double start, double finish, int& iproc, int& nproc, bool print_neighbour=false){
    if (iproc == 0) {
        std::cout << std::endl;
        std::cout << "Number of process : " << nproc << std::endl;
        std::cout << "Total particles : " << neighbour.size() << std::endl;
        std::cout << "Running time : " << finish - start << std::endl;
        std::cout << "Average neighbour : " << average_neighbour(neighbour) << std::endl;
        std::cout << "Maximum neighbour : " << *std::max_element(neighbour.begin(), neighbour.end()) << std::endl;
        std::cout << "Minimum neighbour : " << *std::min_element(neighbour.begin(), neighbour.end()) << std::endl;
        if (print_neighbour == true){
            std::cout << "Particle and number of neighbours : " << std::endl;
            print_vector(neighbour);
        }
    }
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int nproc, iproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    int N1 = 120;
    int N2 = 10549;
    int N3 = 147023;

    double L1 = 18.0;
    double L2 = 100.0;
    double L3 = 200.0;

    double r_c = 9.0;

    std::vector<double> ar_120;
    std::vector<double> ar_10549;
    std::vector<double> ar_147023;

    std::vector<int> neighbour1(N1, 0);
    std::vector<int> neighbour2(N2, 0);
    std::vector<int> neighbour3(N3, 0);

    if (iproc == 0){
        ar_120 = flatten_matrix(read_xyz_file("argon120.xyz", N1, L1));
        ar_10549 = flatten_matrix(read_xyz_file("argon10549.xyz", N2, L2));
        ar_147023 = flatten_matrix(read_xyz_file("argon147023.xyz", N3, L3));   
    }
    
    if (nproc > 1){
        ar_120.resize(N1 * 3);
        ar_10549.resize(N2 * 3);
        ar_147023.resize(N3 * 3);

        MPI_Bcast(ar_120.data(), N1 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ar_10549.data(), N2 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ar_147023.data(), N3 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
 
    double start1 = MPI_Wtime();
    brute_force(ar_120, r_c, neighbour1, iproc, nproc);
    double finish1 = MPI_Wtime();
    write_summary(neighbour1, start1, finish1, iproc, nproc); //Argument true to print out all neighbours

    double start2 = MPI_Wtime();
    brute_force(ar_10549, r_c, neighbour2, iproc, nproc);
    double finish2 = MPI_Wtime();
    write_summary(neighbour2, start2, finish2, iproc, nproc); //Add argument true to print out all neighbours

    double start3 = MPI_Wtime();
    brute_force(ar_147023, r_c, neighbour3, iproc, nproc);
    double finish3 = MPI_Wtime();
    write_summary(neighbour3, start3, finish3, iproc, nproc); //Add argument true to print out all neighbours

    MPI_Finalize();
    return 0;
}