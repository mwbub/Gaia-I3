import warnings
warnings.filterwarnings('ignore')
from mcmillan import *
import galpy
galpy.potential.turn_physical_off(McMillan2017)
import sys

sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')
from main.main_program_cluster import main
  
if __name__ == "__main__":
    main(uniformity_method = "projection", gradient_method = "analytic",
         search_method = "all of local", custom_density = None, custom_samples = None,
         custom_centres = None, custom_potential = McMillan2017)