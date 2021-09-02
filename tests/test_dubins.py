from Utilities import Bundle, cell, zeros, np
from DynamicalSystems import DubinsCar, dubins_default_params

def test_dubins_dyn_sys(params):
        obj = Bundle(dict())
        obj.nx = 3          # Number of state dimensions
        obj.nu = 1          # Number of control inputs
        obj.nd = 3          # Number of disturbance dimensions

        obj.x = None           # State
        obj.u = None          # Recent control signal

        obj.xhist = params['xhist']       # History of state
        obj.uhist = params['uhist']       # History of control

        obj.pdim = params['pdim']        # position dimensions
        obj.vdim = params['vdim']        # velocity dimensions
        obj.hdim = params['hdim']        # heading dimensions

        ## Figure handles
        obj.hpxpy = params['hpxpy']           # Position
        obj.hpxpyhist = params['hpxpyhist']       # Position history
        obj.hvxvy = params['hvxvy']           # Velocity
        obj.hvxvyhist = params['hvxvyhist']       # Velocity history

        # Position velocity (so far only used in DoubleInt)
        obj.hpv = cell(2,1);
        obj.hpvhist = cell(2,1);

        # Data (any data that one may want to store for convenience)
        obj.data = None

        car = DubinsCar(**params)

        print('Dubins car arributes: ')

        for k, v in car.__dict__.items():
            print(f'{k}: {v})' )

        return car


def main():

    x= np.array(([[6, 7, 8, 9, 10, 11, 12]]))
    new_params = dict(x=x, nu=2, nd=6,  speed=5.0)
    print(f'x: {dubins_default_params['x']}\nnu: \'
            '{dubins_default_params['nu']},\nnd: \'
            '{dubins_default_params['nd']}\tspeed: \'
            '{dubins_default_params['speed']}')

    dubins_default_params.update(new_params)

    car = test_dubins_dyn_sys(dubins_default_params)


    print(f'x: {car.x}\nnu: {car.nu},\nnd: {car.nd}\tspeed: {car.speed}')
    print(f'x: {car.x}\nnu: {car.nu},\nnd: {car.nd}\tspeed: {car.speed}')

if __name__ == '__main__':
    main()
