#!/bin/bash

rename -f  'y/A-Z/a-z/' *

cd levelsetpy

rename -f  'y/A-Z/a-z/' *

grep -rl 'LevelSetPy.BoundaryCondition' ./ | xargs sed -i 's/LevelSetPy.BoundaryCondition/levelsetpy.boundarycondition/g'
grep -rl 'LevelSetPy.DynamicalSystems' ./ | xargs sed -i 's/LevelSetPy.DynamicalSystems/levelsetpy.dynamicalsystems/g'
grep -rl 'LevelSetPy.ExplicitIntegration' ./ | xargs sed -i 's/LevelSetPy.ExplicitIntegration/levelsetpy.explicitintegration/g'

grep -rl 'LevelSetPy.ExplicitIntegration.Dissipation' ./ | xargs sed -i 's/LevelSetPy.ExplicitIntegration.Dissipation/levelsetpy.explicitintegration.dissipation/g'
grep -rl 'LevelSetPy.ExplicitIntegration.Integration' ./ | xargs sed -i 's/LevelSetPy.ExplicitIntegration.Integration/levelsetpy.explicitintegration.integration/g'
grep -rl 'LevelSetPy.ExplicitIntegration.Term' ./ | xargs sed -i 's/LevelSetPy.ExplicitIntegration.Term/levelsetpy.explicitintegration.term/g'

grep -rl '.Integration' ./ | xargs sed -i 's/.Integration/.integration/g'

grep -rl 'LevelSetPy.Grids' ./ | xargs sed -i 's/LevelSetPy.Grids/levelsetpy.grids/g'
grep -rl 'LevelSetPy.InitialConditions' ./ | xargs sed -i 's/LevelSetPy.InitialConditions/levelsetpy.initialconditions/g'

grep -rl 'LevelSetPy.SpatialDerivative' ./ | xargs sed -i 's/LevelSetPy.SpatialDerivative/levelsetpy.spatialderivative/g'
cd spatialderivative 
rename -f 'y/A-Z/a-z/'

grep -rl 'LevelSetPy.Utilities' ./ | xargs sed -i 's/LevelSetPy.Utilities/levelsetpy.utilities/g'

cd ..
grep -rl 'LevelSetPy.Utilities' ./ | xargs sed -i 's/LevelSetPy.Utilities/levelsetpy.utilities/g'
grep -rl 'LevelSetPy.ValueFuncs' ./ | xargs sed -i 's/LevelSetPy.ValueFuncs/levelsetpy.valuefuncs/g'
grep -rl 'LevelSetPy.Visualization' ./ | xargs sed -i 's/LevelSetPy.Visualization/levelsetpy.visualization/g'



grep -rl '.term' ./ | xargs sed -i 's/.term/.term/g'
