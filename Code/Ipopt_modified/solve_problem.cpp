// Copyright (C) 2005, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Andreas Waechter            IBM    2004-11-05
//           Zenin Easa Panthakkalakath

// To run this (examples):
// ./solve_problem MBndryCntrlDiri 100 0.01 -1e20 3.5 0. 10. -20. "3.+5.*(x1*(x1-1.)*x2*(x2-1.))"
// ./solve_problem MBndryCntrlNeum 100 0.01 -1e20 2.071 3.7 4.5 4.1 "2.-2.*(x1*(x1-1.)+x2*(x2-1.))"

#include "IpIpoptApplication.hpp"
#include "RegisteredTNLP.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace Ipopt;
using namespace std;

#include "MittelmannBndryCntrlDiri.hpp"
REGISTER_TNLP(MittelmannBndryCntrlDiriCustom, MBndryCntrlDiri)

#include "MittelmannBndryCntrlNeum.hpp"
REGISTER_TNLP(MittelmannBndryCntrlNeumCustom, MBndryCntrlNeum)


static void print_problems()
{
   printf("\nList of all registered problems:\n\n");
   RegisteredTNLPs::PrintRegisteredProblems();
}

int main(
   int   argv,
   char* argc[]
)
{
   if( argv == 2 && !strcmp(argc[1], "list") )
   {
      print_problems();
      return 0;
   }

   if ( argv != 10) {
      // TODO: This should be updated
      printf("Usage: %s (this will ask for problem name)\n", argc[0]);
      printf("       %s ProblemName N alpha lb_y ub_y lb_u ub_u d_const_OR_u_init target_profile_equation\n", argc[0]);
      printf("          where ... TODO: Describe the params ... \n");
      printf("       %s list\n", argc[0]);
      printf("          to list all registered problems.\n");
      return -1;
   }

   // Now, we have 10 arguments:
   // fileBeingRun, problemName, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const_OR_u_init, target_profile_equation

   SmartPtr<RegisteredTNLP> tnlp;
   Index N;
   Number alpha, lb_y, ub_y, lb_u, ub_u, d_const_OR_u_init;
   std::string target_profile_equation;

   N = std::stoi(argc[2]);
   alpha = std::stod(argc[3]);
   lb_y = std::stod(argc[4]);
   ub_y = std::stod(argc[5]);
   lb_u = std::stod(argc[6]);
   ub_u = std::stod(argc[7]);
   d_const_OR_u_init = std::stod(argc[8]);
   target_profile_equation = argc[9];

   std::cout << "Given inputs: \n";
   std::cout << N << "\t" << alpha << "\t" << lb_y << "\t" << ub_y << "\t" << lb_u << "\t" << ub_u
      << "\t" << d_const_OR_u_init << "\t" << target_profile_equation << "\n";


#ifdef TIME_LIMIT
   int runtime;
   if( argv == 4 )
   {
      runtime = atoi(argc[3]);
   }
   else
#endif

   // Create an instance of your nlp...
   tnlp = RegisteredTNLPs::GetTNLP(argc[1]);
   if( !IsValid(tnlp) )
   {
      printf("Problem with name \"%s\" not known.\n", argc[1]);
      print_problems();
      return -2;
   }

   bool retval = tnlp->InitializeProblem(
      N, alpha, lb_y, ub_y, lb_u, ub_u, d_const_OR_u_init,
      target_profile_equation
   );
   if( !retval )
   {
      printf("Cannot initialize problem.  Abort.\n");
      return -4;
   }

   // Create an instance of the IpoptApplication
   // We are using the factory, since this allows us to compile this
   // example with an Ipopt Windows DLL
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   ApplicationReturnStatus status;
   status = app->Initialize();
   if( status != Solve_Succeeded )
   {
      printf("\n\n*** Error during initialization!\n");
      return (int) status;
   }
   // Set option to use internal scaling
   // DOES NOT WORK FOR VLUKL* PROBLEMS:
   // app->Options()->SetStringValueIfUnset("nlp_scaling_method", "user-scaling");

#ifdef TIME_LIMIT
   app->Options()->SetNumericValue("max_wall_time", runtime);
#endif

   status = app->OptimizeTNLP(GetRawPtr(tnlp));

   return (int) status;
}
