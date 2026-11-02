import numpy as np
import pytest
try:
    from .solver import Coefficient, inverse, inverse_gradient, residual, source, solve_periodic, gaussian_duhamel
    from .benchmark import varying_coefficient, reference
except ImportError:  # direct pytest collection from this project root
    from solver import Coefficient, inverse, inverse_gradient, residual, source, solve_periodic, gaussian_duhamel
    from benchmark import varying_coefficient, reference


def test_chain_rule_source_with_time_and_space_varying_coefficient():
    x = np.linspace(0, 2*np.pi, 101)
    t, delta, a = .3, .4, .7
    v = reference(t, x)
    p = np.exp(-delta*t/2)*np.cos(x-a*t)
    vxx = -v
    vt = -delta*v/2-a*p
    c = varying_coefficient(t, x)
    omega = np.exp(-c.value*v)
    wx = -omega*(c.gradient*v+c.value*p)
    wt = -omega*(c.time*v+c.value*vt)
    wxx = omega*((c.gradient*v+c.value*p)**2
                 -(c.laplacian*v+2*c.gradient*p+c.value*vxx))
    actual = source(t, x, omega, wx, c, lambda t,x,p: a*p, delta)
    np.testing.assert_allclose(actual, wt-delta*wxx/2, atol=3e-15)
    np.testing.assert_allclose(inverse_gradient(omega, wx, c.value,c.gradient),p,atol=1e-15)


def test_quadratic_limit_is_heat_semigroup():
    delta=.4
    def c(t,x):
        return Coefficient(np.full_like(x,1/delta), np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
    g=lambda x: -delta*np.log(1+.2*np.cos(x))
    out=solve_periodic(g,c,lambda t,x,p:.5*p*p,steps=10)
    expected=1+.2*np.exp(-delta*.5/2)*np.cos(out["x"])
    np.testing.assert_allclose(out["omega"],expected,atol=2e-15)


def test_full_source_refinement_and_dropped_source_bias():
    errors=[]
    for steps in (100,200,400):
        out=solve_periodic(np.sin,varying_coefficient,lambda t,x,p:.7*p,steps=steps)
        errors.append(np.max(np.abs(out["value"]-reference(.5,out["x"]))))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2
    assert errors[-1] < .001
    dropped=solve_periodic(np.sin,varying_coefficient,lambda t,x,p:.7*p,steps=400,mode="drop_derivative")
    assert np.max(np.abs(dropped["value"]-reference(.5,dropped["x"]))) > 10*errors[-1]


@pytest.mark.parametrize("omega,c",[(0.,1.),(-1.,1.),(np.nan,1.),(1.,0.),(1.,-1.),(1.,np.inf)])
def test_invalid_inverse_is_rejected(omega,c):
    with pytest.raises(ValueError): inverse(omega,c)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        source(0,0,np.array([1.]),np.array([0.]),Coefficient(1,0,0,0),lambda t,x,p:p,.4,"unknown")


def test_gaussian_duhamel_polynomial_source():
    # S_t(y^2)=x^2+delta*t. Integral S_(t-s)(y)=t*x.
    x,t,delta=.4,.6,.3
    out=gaussian_duhamel(x,t,delta,lambda y:y*y+1,lambda s,y:y,
                         samples=300000,seed=1234)
    expected=1+x*x+delta*t+t*x
    expected_grad=2*x+t
    assert abs(out["omega"]-expected) < 5*out["omega_se"]
    assert abs(out["gradient"]-expected_grad) < 5*out["gradient_se"]


def test_coefficient_zero_gradient_no_singularity():
    c=Coefficient(np.array([1.]),0.,0.,0.)
    alg,der=residual(np.array([.2]),np.array([0.]),np.array([0.]),c,.4)
    assert alg[0]==0 and der[0]==0
