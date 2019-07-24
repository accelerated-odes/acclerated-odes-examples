#include <iostream>

#include "SdcIntegrator.H"
#include "SparseGaussJordan.H"
#include "vode_system.H"
#include "RealVector.H"
#include "WallTimer.H"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CVODE
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#ifdef USE_KLU
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#else
#include <sunlinsol/sunlinsol_spgmr.h>
#endif
#endif

template<class SparseLinearSolver, class SystemClass, size_t order>
void do_sdc_host(Real* y_initial, Real* y_final, 
		 Real start_time, Real end_time, Real start_timestep,
		 Real tolerance, size_t maximum_newton_iters, 
		 bool fail_if_maximum_newton, Real maximum_steps,
		 Real epsilon, size_t size, bool use_adaptive_timestep) {

  typedef SdcIntegrator<SparseLinearSolver,SystemClass,order> SdcIntClass;

  SystemClass ode_system;

  for (size_t global_index = 0; global_index < size; global_index++) {
    SdcIntClass sdc;
    RealVector<SystemClass::neqs> y_ini;

    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_ini.data[i] = y_initial[global_index * SystemClass::neqs + i];
    }

    SdcIntClass::set_jacobian_layout(sdc, ode_system);
    SdcIntClass::initialize(sdc, y_ini, 
			    start_time, end_time, start_timestep,
			    tolerance, maximum_newton_iters, 
			    fail_if_maximum_newton, maximum_steps,
			    epsilon, use_adaptive_timestep);

    for (size_t i = 0; i < maximum_steps; i++) {
      SdcIntClass::prepare(sdc);
      SdcIntClass::solve(sdc);
      SdcIntClass::update(sdc);
      if (SdcIntClass::is_finished(sdc)) break;
    }

    RealVector<SystemClass::neqs>& y_fin = SdcIntClass::get_current_solution(sdc);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_final[global_index * SystemClass::neqs + i] = y_fin.data[i];
    }
  }
}

#ifdef USE_CVODE
int cv_rhs(Real t, N_Vector y, N_Vector ydot, void *user_data)
{
  // TODO: need a way to set the underlying data of a RealVector
  // so we can wrap the data pointer in a RealVector before calling
  // VodeSystem::evaluate(t, yvec, rhsvec).
  Real *ydata = N_VGetArrayPointer(y);
  Real *rhsdata = N_VGetArrayPointer(ydot);
  rhsdata[0] = -0.04 * ydata[0] + 1.e4 * ydata[1] * ydata[2];
  rhsdata[1] =  0.04 * ydata[0] - 1.e4 * ydata[1] * ydata[2] - 3.e7 * ydata[1] * ydata[1];
  rhsdata[2] =  3.e7 * ydata[1] * ydata[1];
  return 0;
}

#ifdef USE_KLU
int cv_jac(Real t, N_Vector y, N_Vector f_y, SUNMatrix Jac, void *user_data,
  N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // TODO: need a way to set the underlying data of a RealVector/RealSparseMatrix
  // so we can wrap the data pointers before calling
  // VodeSystem::evaluate(t, yvec, jac).
  Real *ydata = N_VGetArrayPointer(y);
  Real *rhsdata = N_VGetArrayPointer(f_y);
  Real *jacdata = SUNSparseMatrix_Data(Jac);
  rhsdata[0] = -0.04 * ydata[0] + 1.e4 * ydata[1] * ydata[2];
  rhsdata[1] =  0.04 * ydata[0] - 1.e4 * ydata[1] * ydata[2] - 3.e7 * ydata[1] * ydata[1];
  rhsdata[2] =  3.e7 * ydata[1] * ydata[1];

  jacdata[0] = -0.04e0;
  jacdata[1] =  1.e4 * ydata[2];
  jacdata[2] =  1.e4 * ydata[1];

  jacdata[3] =  0.04e0;
  jacdata[4] = -1.e4 * ydata[2] - 6.e7 * ydata[1];
  jacdata[5] = -1.e4 * ydata[1];

  jacdata[6] = 6.e7 * ydata[1];
  jacdata[7] = 0.0e0;
  return 0;
}
#endif

template<class SystemClass, size_t order>
void do_cvode(Real* y_initial, Real* y_final, 
		 Real start_time, Real end_time, Real start_timestep,
		 Real tolerance, size_t maximum_newton_iters, 
		 bool fail_if_maximum_newton, Real maximum_steps,
		 Real epsilon, size_t size, bool use_adaptive_timestep) {

  int retval;
  void *cvode_mem;
  N_Vector yi, yf;
  SUNMatrix A;
  SUNLinearSolver LS;
  Real t;

  yi = NULL;
  yf = NULL;
  A = NULL;
  LS = NULL;
  cvode_mem = NULL;

#ifdef USE_OPENMP
  #pragma omp parallel for private(cvode_mem, yi, yf, A, LS, t, retval)
#endif
  for (size_t global_index = 0; global_index < size; global_index++) {

    yi = N_VMake_Serial(SystemClass::neqs,
      &y_initial[global_index * SystemClass::neqs]);
    yf = N_VMake_Serial(SystemClass::neqs,
      &y_final[global_index * SystemClass::neqs]);

    cvode_mem = CVodeCreate(CV_BDF);
    retval = CVodeInit(cvode_mem, cv_rhs, start_time, yi);
    retval = CVodeSStolerances(cvode_mem, tolerance, tolerance);
    retval = CVodeSetMaxNumSteps(cvode_mem, maximum_steps);
    if (!use_adaptive_timestep) {
      retval = CVodeSetFixedOrd(cvode_mem, order);
      retval = CVodeSetFixedStep(cvode_mem, start_timestep);
    }

#ifdef USE_KLU
    A = SUNSparseMatrix(SystemClass::neqs, SystemClass::neqs, SystemClass::nnz, CSR_MAT);  
    LS = SUNLinSol_KLU(yi, A);
    retval = CVodeSetLinearSolver(cvode_mem, LS, A); 
    retval = CVodeSetJacFn(cvode_mem, cv_jac);
#else
    LS = SUNLinSol_SPGMR(yi, PREC_NONE, 0);
    retval = CVodeSetLinearSolver(cvode_mem, LS, NULL); 
#endif

    retval = CVode(cvode_mem, end_time, yf, &t, CV_NORMAL);

    N_VDestroy(yi);
    N_VDestroy(yf);
#ifdef USE_KLU
    SUNMatDestroy(A);
#endif
    SUNLinSolFree(LS);
    CVodeFree(&cvode_mem);

    if (retval != CV_SUCCESS)
      std::cout << "ERROR: CVode returned " << retval << std::endl;

  }

}
#endif

int main(int argc, char* argv[]) {

  size_t grid_size = 32;

  size_t num_systems = grid_size * grid_size * grid_size;

  const size_t order = 3; // order 4 causes CVODE to error out unless timestep is reduced

  WallTimer timer;

  Real* y_initial;
  Real* y_final;

  y_initial = new Real[VodeSystem::neqs * num_systems];
  y_final = new Real[VodeSystem::neqs * num_systems];

  // initialize systems
  for (size_t i = 0; i < num_systems; i += VodeSystem::neqs) {
    y_initial[i] = 1.0;
    y_initial[i+1] = 0.0;
    y_initial[i+2] = 0.0;
  }

  Real start_time = 0.0;
  Real end_time = 1.0;
  Real start_timestep = (end_time - start_time)/5120.0;
  Real tolerance = 1.0e-12;
  size_t maximum_newton_iters = 1000;
  size_t maximum_steps = 1000000;
  bool fail_if_maximum_newton = true;
  Real epsilon = std::numeric_limits<Real>::epsilon();
  bool use_adaptive_timestep = false;

  std::cout << "Starting integration ..." << std::endl;

  timer.start_wallclock();

#ifdef USE_CVODE
  do_cvode<VodeSystem, order>(y_initial, y_final, start_time, end_time,
    start_timestep, tolerance, maximum_newton_iters, fail_if_maximum_newton,
    maximum_steps, epsilon, num_systems, use_adaptive_timestep);
#else
  do_sdc_host<SparseGaussJordan, VodeSystem, order>(y_initial, y_final,
						    start_time, end_time, start_timestep,
						    tolerance, maximum_newton_iters,
						    fail_if_maximum_newton, maximum_steps,
						    epsilon, num_systems, use_adaptive_timestep);
#endif

  timer.stop_wallclock();

  std::cout << std::endl << "Final Integration States -------------------" << std::endl;
  for (size_t i = 0; i < num_systems; i += VodeSystem::neqs) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_final[" << i << "]: " << std::endl;
    std::cout << " ";
    for (size_t j = 0; j < VodeSystem::neqs; j++) {
      std::cout << y_final[i + j] << " ";
    }
    std::cout << std::endl;
  }


  std::cout << "Finished execution on host CPU" << std::endl;
  std::cout << std::endl << "Integration walltime (s): " << timer.get_walltime() << std::endl;

#ifdef USE_CVODE
  std::cout << "Integrator: CVODE" << std::endl;
#ifdef USE_KLU
  std::cout << "Linear Solver: KLU" << std::endl;
#else
  std::cout << "Linear Solver: SPGMR" << std::endl;
#endif
#else
  std::cout << "Integrator: SDC" << std::endl;
#endif

  delete[] y_initial;
  delete[] y_final;

  return 0;
}
