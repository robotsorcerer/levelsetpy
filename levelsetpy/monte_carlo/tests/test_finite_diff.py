"""Independent Fourier-mode and manufactured checks for local full-source solves."""
import numpy as np
import pytest
from hjgauss.core.finite_diff import solve,original_reference,sixth_derivatives
from hjgauss.validation.cases import manufactured


@pytest.mark.parametrize('dimension',[2,3])
def test_full_fd4_spatial_refinement_with_variable_coefficient(dimension):
    c=manufactured(dimension);errors=[]
    for n in [24,48]:
        sol=solve(c['initial'],c['hamiltonian'],bounds=c['bounds'],shape=(n,)*dimension,
                  delta=c['delta'],horizon=c['horizon'],steps=160)
        x=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
        errors.append(float(np.max(np.abs(sol['value']-c['exact'](.4,x)[0]))))
    assert errors[1]<.09*errors[0]
    assert errors[1]<1e-5


def test_full_fd4_heat_limit_has_fourth_order_time_refinement():
    n=8;h=2*np.pi/n;delta=.3;c=.2;horizon=.4
    coefficient=lambda t,x:(c,0.,(0.,0.),0.)
    initial=lambda x:-np.log(1+.1*np.cos(2*x[0])*np.cos(2*x[1]))/c
    H=lambda t,x,p:.5*delta*c*sum(q*q for q in p)
    lam=.5*delta*2*(-2*np.cos(4*h)+32*np.cos(2*h)-30)/(12*h*h)
    errors=[]
    for steps in [4,8]:
        sol=solve(initial,H,bounds=((-np.pi,np.pi),)*2,shape=(n,n),delta=delta,
                  horizon=horizon,steps=steps,coefficient=coefficient)
        x,y=np.meshgrid(*sol['axes'],indexing='ij')
        exact=-np.log(1+.1*np.exp(lam*horizon)*np.cos(2*x)*np.cos(2*y))/c
        errors.append(float(np.max(np.abs(sol['value']-exact))))
    assert errors[1]<.09*errors[0]


@pytest.mark.parametrize('dimension',[2,3])
def test_fd6_original_reference_against_exact_manufactured_solution(dimension):
    c=manufactured(dimension);n=32
    sol=original_reference(c['initial'],c['hamiltonian'],bounds=c['bounds'],shape=(n,)*dimension,
                           delta=c['delta'],horizon=c['horizon'],steps=160)
    x=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
    assert np.max(np.abs(sol['value']-c['exact'](.4,x)[0]))<1e-6
