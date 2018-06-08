import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search import search_local
import time

def get_load_state():
    return ('_GAIA_LOADED = {}\n_PARALLAX_CUT = {}\n'
            ).format(search_local._GAIA_LOADED, search_local._PARALLAX_CUT)
    
def run_search(u0, v0, w0, U0, V0, W0, epsilon, v_scale, parallax_cut):
    print('Initial load state:')
    print(get_load_state())
    print('Running with parallax_cut = {}\n'.format(parallax_cut))
    start = time.process_time()
    results = search_local.search_phase_space(u0, v0, w0, U0, V0, W0, 
                                              epsilon, v_scale, 
                                              parallax_cut=parallax_cut)
    end = time.process_time()
    elapsed = end - start
    print('Time elapsed: {} s'.format(elapsed))
    print('Number of results: {}'.format(len(results)))
    print('New load state:')
    print(get_load_state())

def test_search_phase_space(u0, v0, w0, U0, V0, W0, epsilon, v_scale):
    for i in range(3):
        run_search(u0, v0, w0, U0, V0, W0, epsilon, v_scale, True)
        
    for i in range(3):
        run_search(u0, v0, w0, U0, V0, W0, epsilon, v_scale, False)
    
    