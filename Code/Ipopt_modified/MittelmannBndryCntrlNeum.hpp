// Copyright (C) 2005, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Andreas Waechter           IBM    2005-10-18
//                  based on MyNLP.hpp

#ifndef __MITTELMANNBNDRYCNTRLNEUM_HPP__
#define __MITTELMANNBNDRYCNTRLNEUM_HPP__

#include "IpTNLP.hpp"
#include "RegisteredTNLP.hpp"
#include "MathParser.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace Ipopt;

/** Base class for boundary control problems with Neumann boundary
 *  conditions, as formulated by Hans Mittelmann as Examples 5-8 in
 *  "Optimization Techniques for Solving Elliptic Control Problems
 *  with Control and State Constraints. Part 1: Boundary Control"
 */
class MittelmannBndryCntrlNeumBase: public RegisteredTNLP
{
public:
   /** Constructor.  N is the number of mesh points in one dimension
    *  (excluding boundary). */
   MittelmannBndryCntrlNeumBase();

   /** Default destructor */
   virtual ~MittelmannBndryCntrlNeumBase();

   /**@name Overloaded from TNLP */
   //@{
   /** Method to return some info about the nlp */
   virtual bool get_nlp_info(
      Index&          n,
      Index&          m,
      Index&          nnz_jac_g,
      Index&          nnz_h_lag,
      IndexStyleEnum& index_style
   );

   /** Method to return the bounds for my problem */
   virtual bool get_bounds_info(
      Index   n,
      Number* x_l,
      Number* x_u,
      Index   m,
      Number* g_l,
      Number* g_u
   );

   /** Method to return the starting point for the algorithm */
   virtual bool get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda
   );

   /** Method to return the objective value */
   virtual bool eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   );

   /** Method to return the gradient of the objective */
   virtual bool eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   );

   /** Method to return the constraint residuals */
   virtual bool eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       g
   );

   /** Method to return:
    *   1) The structure of the Jacobian (if "values" is NULL)
    *   2) The values of the Jacobian (if "values" is not NULL)
    */
   virtual bool eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nele_jac,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** Method to return:
    *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
    *   2) The values of the Hessian of the Lagrangian (if "values" is not NULL)
    */
   virtual bool eval_h(
      Index         n,
      const Number* x,
      bool          new_x,
      Number        obj_factor,
      Index         m,
      const Number* lambda,
      bool          new_lambda,
      Index         nele_hess,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** Method for returning scaling parameters */
   virtual bool get_scaling_parameters(
      Number& obj_scaling,
      bool&   use_x_scaling,
      Index   n,
      Number* x_scaling,
      bool&   use_g_scaling,
      Index   m,
      Number* g_scaling
   );

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   virtual void finalize_solution(
      SolverReturn               status,
      Index                      n,
      const Number*              x,
      const Number*              z_L,
      const Number*              z_U,
      Index                      m,
      const Number*              g,
      const Number*              lambda,
      Number                     obj_value,
      const IpoptData*           ip_data,
      IpoptCalculatedQuantities* ip_cq
   );
   //@}

protected:
   /** Method for setting the internal parameters that define the problem.
    *
    *  It must be called by the child class in its
    *  implementation of InitializeParameters
    */
   void SetBaseParameters(
      Index  N,
      Number alpha,
      Number lb_y,
      Number ub_y,
      Number lb_u,
      Number ub_u,
      Number u_init);

   /**@name Functions that defines a particular instance. */
   //@{
   /** Target profile function for y (and initial guess function) */
   virtual Number y_d_cont(
      Number x1,
      Number x2
   ) const = 0;

   /** Forcing function for the elliptic equation */
   virtual Number d_cont(
      Number x1,
      Number x2,
      Number y
   ) const = 0;

   /** First partial derivative of forcing function w.r.t. y */
   virtual Number d_cont_dy(
      Number x1,
      Number x2,
      Number y
   ) const = 0;

   /** Second partial derivative of forcing function w.r.t. y,y */
   virtual Number d_cont_dydy(
      Number x1,
      Number x2,
      Number y
   ) const = 0;

   /** returns true if second partial derivative of d_cont
    *  w.r.t. y,y is always zero.
    */
   virtual bool d_cont_dydy_alwayszero() const = 0;

   /** Function in Neuman boundary condition */
   virtual Number b_cont(
      Number x1,
      Number x2,
      Number y,
      Number u
   ) const = 0;

   /** First partial derivative of b_cont w.r.t. y */
   virtual Number b_cont_dy(
      Number x1,
      Number x2,
      Number y,
      Number u
   ) const = 0;

   /** First partial derivative of b_cont w.r.t. u */
   virtual Number b_cont_du(
      Number x1,
      Number x2,
      Number y,
      Number u
   ) const = 0;

   /** Second partial derivative of b_cont w.r.t. y,y */
   virtual Number b_cont_dydy(
      Number x1,
      Number x2,
      Number y,
      Number u
   ) const = 0;

   /** returns true if second partial derivative of b_cont
    *  w.r.t. y,y is always zero.
    */
   virtual bool b_cont_dydy_alwayszero() const = 0;
   //@}

private:
   /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
   //@{
   MittelmannBndryCntrlNeumBase(
      const MittelmannBndryCntrlNeumBase&
   );

   MittelmannBndryCntrlNeumBase& operator=(
      const MittelmannBndryCntrlNeumBase&
   );
   //@}

   /**@name Problem specification */
   //@{
   /** Number of mesh points in one dimension (excluding boundary) */
   Index N_;
   /** Step size */
   Number h_;
   /** h_ squaredd */
   Number hh_;
   /** overall lower bound on y */
   Number lb_y_;
   /** overall upper bound on y */
   Number ub_y_;
   /** overall lower bound on u */
   Number lb_u_;
   /** overall upper bound on u */
   Number ub_u_;
   /** Initial value for the controls u */
   Number u_init_;
   /** Weighting parameter for the control target deviation functional
    *  in the objective
    */
   Number alpha_;
   /** Array for the target profile for y */
   Number* y_d_;
   //@}

   /**@name Auxiliary methods */
   //@{
   /** Translation of mesh point indices to NLP variable indices for y(x_ij) */
   inline Index y_index(
      Index i,
      Index j
   ) const
   {
      return j + (N_ + 2) * i;
   }

   /** Translation of mesh point indices to NLP variable indices for
    *  u(x_ij) on {0} x (0,1) boundary
    */
   inline Index u0j_index(
      Index j
   ) const
   {
      return (N_ + 2) * (N_ + 2) + j - 1;
   }

   /** Translation of mesh point indices to NLP variable indices for
    *  u(x_ij) on {1} x (0,1) boundary
    */
   inline Index u1j_index(
      Index j
   ) const
   {
      return (N_ + 2) * (N_ + 2) + N_ + j - 1;
   }

   /** Translation of mesh point indices to NLP variable indices for
    *  u(x_ij) on (0,1) x {0} boundary
    */
   inline Index ui0_index(
      Index j
   ) const
   {
      return (N_ + 2) * (N_ + 2) + 2 * N_ + j - 1;
   }

   /** Translation of mesh point indices to NLP variable indices for
    *  u(x_ij) on (0,1) x {1} boundary
    */
   inline Index ui1_index(
      Index j
   ) const
   {
      return (N_ + 2) * (N_ + 2) + 3 * N_ + j - 1;
   }

   /** Compute the grid coordinate for given index in x1 direction */
   inline Number x1_grid(
      Index i
   ) const
   {
      return h_ * (Number) i;
   }

   /** Compute the grid coordinate for given index in x2 direction */
   inline Number x2_grid(
      Index j
   ) const
   {
      return h_ * (Number) j;
   }
   //@}
};

/** Implementation of Example Custom */
class MittelmannBndryCntrlNeumCustom: public MittelmannBndryCntrlNeumBase
{
public:
   MittelmannBndryCntrlNeumCustom()
   { }

   virtual ~MittelmannBndryCntrlNeumCustom()
   { }

   virtual bool InitializeProblem(
      Index N,
      Number alpha,
      Number lb_y,
      Number ub_y,
      Number lb_u,
      Number ub_u,
      Number u_init,
      std::string target_profile_equation
   )
   {
      if( N < 1 )
      {
         printf("N has to be at least 1.");
         return false;
      }

      mTarget_profile_equation = target_profile_equation;
      SetBaseParameters(N, alpha, lb_y, ub_y, lb_u, ub_u, u_init);
      return true;
   }

protected:
   /** Target profile function for y */
   virtual Number y_d_cont(
      Number x1,
      Number x2
   ) const
   {
      MathParser::parser ob;

      std::string expression = mTarget_profile_equation;
      ob.replace_in_exp(expression, "x1", std::to_string(x1));
      ob.replace_in_exp(expression, "x2", std::to_string(x2));

      double ans = ob.eval_exp(expression);
      if (*ob.errormsg)
         throw ob.errormsg;

      return ans;
      //return 2. - 2. * (x1 * (x1 - 1.) + x2 * (x2 - 1.));
   }

   /** Forcing function for the elliptic equation */
   virtual Number d_cont(
      Number /*x1*/,
      Number /*x2*/,
      Number /*y*/
   ) const
   {
      return 0.;
   }

   /** First partial derivative of forcing function w.r.t. y */
   virtual Number d_cont_dy(
      Number /*x1*/,
      Number /*x2*/,
      Number /*y*/
   ) const
   {
      return 0.;
   }

   /** Second partial derivative of forcing function w.r.t y,y */
   virtual Number d_cont_dydy(
      Number /*x1*/,
      Number /*x2*/,
      Number /*y*/
   ) const
   {
      return 0.;
   }

   /** returns true if second partial derivative of d_cont
    *  w.r.t. y,y is always zero.
    */
   virtual bool d_cont_dydy_alwayszero() const
   {
      return true;
   }

   /** Function in Neuman boundary condition */
   virtual Number b_cont(
      Number /*x1*/,
      Number /*x2*/,
      Number y,
      Number u
   ) const
   {
      return u - y * y;
   }

   /** First partial derivative of b_cont w.r.t. y */
   virtual Number b_cont_dy(
      Number /*x1*/,
      Number /*x2*/,
      Number y,
      Number /*u*/
   ) const
   {
      return -2. * y;
   }

   /** First partial derivative of b_cont w.r.t. u */
   virtual Number b_cont_du(
      Number /*x1*/,
      Number /*x2*/,
      Number /*y*/,
      Number /*u*/
   ) const
   {
      return 1.;
   }

   /** Second partial derivative of b_cont w.r.t. y,y */
   virtual Number b_cont_dydy(
      Number /*x1*/,
      Number /*x2*/,
      Number /*y*/,
      Number /*u*/
   ) const
   {
      return -2.;
   }

   /** returns true if second partial derivative of b_cont
    *  w.r.t. y,y is always zero.
    */
   virtual bool b_cont_dydy_alwayszero() const
   {
      return false;
   }

private:
   /**@name hide implicitly defined constructors and copy operators */
   //@{
   MittelmannBndryCntrlNeumCustom(
      const MittelmannBndryCntrlNeumCustom&
   );

   MittelmannBndryCntrlNeumCustom& operator=(
      const MittelmannBndryCntrlNeumCustom&
   );
   //@}

   std::string mTarget_profile_equation;
};

#endif
