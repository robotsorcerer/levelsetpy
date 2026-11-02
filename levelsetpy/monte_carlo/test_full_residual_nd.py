"""Independent chain-rule identities and multidimensional refinement checks."""
import numpy as np
import pytest
from full_residual_nd import mesh, variable_coefficient, full_source, solve, spectral_derivatives
from multidimensional_validation import manufactured, game_case


@pytest.mark.parametrize('dimension',[2,3])
def test_even_axis_nyquist_first_derivative_and_laplacian(dimension):
    _,x,_,waves=mesh(((-np.pi,np.pi),)*dimension,(16,)*dimension)
    u=np.broadcast_to(np.cos(8*x[0])*np.sin(x[-1]),(16,)*dimension).copy()
    grad,lap=spectral_derivatives(u,waves)
    np.testing.assert_allclose(grad[0],0,atol=2e-13)
    truth=np.broadcast_to(np.cos(8*x[0])*np.cos(x[-1]),u.shape)
    np.testing.assert_allclose(grad[-1],truth,atol=2e-13)
    np.testing.assert_allclose(lap,-65*u,atol=2e-12)


@pytest.mark.parametrize('dimension',[2,3])
def test_nd_full_chain_rule_with_time_and_space_dependent_coefficient(dimension):
    bounds=((-np.pi,np.pi),)*dimension
    _,x,_,_=mesh(bounds,(12,)*dimension)
    t=.37;delta=.3;cfun=variable_coefficient(bounds)
    ss=[np.sin(q) for q in x];cs=[np.cos(q) for q in x]
    osc=np.exp(-t)*np.prod(np.broadcast_arrays(*ss),axis=0)
    v=.3+osc;vt=-osc;lapv=-dimension*osc;p=[]
    for i in range(dimension):
        q=np.exp(-t)*cs[i]
        for j in range(dimension):
            if j!=i:q=q*ss[j]
        p.append(q)
    c,ct,dc,lapc=cfun(t,x)
    z=np.exp(-c*v)
    dz=tuple(-z*(d*v+c*q) for d,q in zip(dc,p))
    zt=-z*(ct*v+c*vt)
    lapz=z*(sum((d*v+c*q)**2 for d,q in zip(dc,p))
            -lapc*v-2*sum(d*q for d,q in zip(dc,p))-c*lapv)
    h=lambda t,x,p:.5*delta*lapv-vt
    np.testing.assert_allclose(full_source(t,z,dz,x,cfun,h,delta),
                               zt-.5*delta*lapz,atol=2e-15,rtol=2e-14)


@pytest.mark.parametrize('dimension',[2,3])
def test_full_nd_manufactured_time_refinement(dimension):
    case=manufactured(dimension)
    errors=[]
    for steps in [40,80]:
        sol=solve(case['initial'],case['hamiltonian'],bounds=case['bounds'],
                  shape=(24,)*dimension,delta=.3,horizon=.4,steps=steps)
        x=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
        errors.append(np.max(np.abs(sol['value']-case['exact'](.4,x)[0])))
    assert errors[1] < .35*errors[0]
    assert errors[1] < 2e-6


def test_independent_reference_refines_against_analytic_solution():
    case=manufactured(2);errors=[]
    for n in [16,32]:
        sol=solve(case['initial'],case['hamiltonian'],bounds=case['bounds'],
                  shape=(n,n),delta=.3,horizon=.4,steps=80,kind='reference')
        x=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
        errors.append(np.max(np.abs(sol['value']-case['exact'](.4,x)[0])))
    assert errors[1] < .1*errors[0]


def test_invalid_time_steps_and_snapshot_requests_are_rejected():
    case=manufactured(2)
    kwargs=dict(bounds=case['bounds'],shape=(16,16),delta=.3,horizon=.4)
    for steps,snapshots in [(0,()),(20,(-.1,)),(20,(.5,)),(20,(.001,.002))]:
        with pytest.raises(ValueError):
            solve(case['initial'],case['hamiltonian'],steps=steps,
                  snapshots=snapshots,**kwargs)


@pytest.mark.parametrize('name',['double_integrator','dubins_single','dubins_relative','rockets'])
def test_elapsed_game_hamiltonian_matches_enumerated_physical_controls(name):
    rng=np.random.default_rng(67412)
    for _ in range(12):
        x=rng.uniform(-2,2,3);p=rng.normal(size=3)
        case=game_case(name)
        if name=='double_integrator':
            physical=min(p[0]*x[1]+p[1]*u for u in [-1,1])
            elapsed=case['hamiltonian'](0,x[:2],p[:2])
        elif name=='dubins_single':
            physical=min(p[0]*np.cos(x[2])+p[1]*np.sin(x[2])+p[2]*u for u in [-1,1])
            elapsed=case['hamiltonian'](0,x,p)
        else:
            values=[]
            for ue in [-1,1]:
                candidates=[]
                for up in [-1,1]:
                    if name=='rockets':
                        f=[np.cos(x[2])+ue*x[0],np.sin(x[2])+1+up*x[0]-2,up-ue]
                    else:
                        f=[np.cos(x[2])-1+ue*x[1],np.sin(x[2])-ue*x[0],up-ue]
                    candidates.append(np.dot(p,f))
                values.append(min(candidates))
            physical=max(values)
            elapsed=case['hamiltonian'](0,x,p)
        np.testing.assert_allclose(elapsed,-physical,atol=2e-15)


@pytest.mark.parametrize('name',['double_integrator','dubins_single','dubins_relative','rockets','murmuration'])
def test_game_hamiltonian_preserves_constant_payoff(name):
    case=game_case(name)
    d=len(case['bounds']);x=np.ones(d);p=np.zeros(d)
    assert case['hamiltonian'](0,x,p)==0


def test_attacked_four_dimensional_model_reduces_with_equal_vertical_bounds():
    import itertools
    rng=np.random.default_rng(781);H=game_case('murmuration')['hamiltonian']
    for _ in range(12):
        x=rng.uniform(-2,2,4);p=rng.normal(size=4)
        def optimized(gamma_e,gamma_p):
            evader_values=[]
            for we,ze in itertools.product([-1,1],repeat=2):
                pursuer_values=[]
                for wp,zp in itertools.product([-1,1],repeat=2):
                    f=[np.cos(x[3])-1+we*x[1],np.sin(x[3])-we*x[0],
                       gamma_p*zp-gamma_e*ze,wp-we]
                    pursuer_values.append(np.dot(p,f))
                evader_values.append(min(pursuer_values))
            return -max(evader_values)
        reduced=H(0,x[[0,1,3]],p[[0,1,3]])
        np.testing.assert_allclose(optimized(.5,.5),reduced,atol=3e-15)
        np.testing.assert_allclose(optimized(.8,.3),reduced-(.8-.3)*abs(p[2]),atol=3e-15)
