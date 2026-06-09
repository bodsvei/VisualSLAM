"""
check_g2o.py
------------
Run this on your Mac to see exactly which g2o API is available.
Copy the output and it will tell us which solver line to use.

Usage:  python3 check_g2o.py
"""
try:
    import g2o
except ImportError:
    print("g2o is NOT installed.")
    print("Install: pip install g2o-python")
    exit(1)

print(f"g2o file   : {g2o.__file__}")
print(f"g2o version: {getattr(g2o, '__version__', 'no __version__ attr')}")
print()

all_names = sorted(dir(g2o))

# Group by category
categories = {
    "Solvers (Linear)": [n for n in all_names if 'LinearSolver' in n],
    "Solvers (Block)":  [n for n in all_names if 'BlockSolver' in n],
    "Algorithms":       [n for n in all_names if 'Algorithm' in n or 'Levenberg' in n or 'Gauss' in n],
    "Vertices":         [n for n in all_names if n.startswith('Vertex')],
    "Edges":            [n for n in all_names if n.startswith('Edge')],
    "SE3 types":        [n for n in all_names if 'SE3' in n or 'Sim3' in n or 'Quat' in n],
}

for cat, names in categories.items():
    if names:
        print(f"── {cat} {'─'*(40-len(cat))}")
        for n in names:
            print(f"   g2o.{n}")
        print()

# Quick solver probe
print("── Solver probe ─────────────────────────────")
solver_attempts = [
    ("Cholmod SE3",  "g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())"),
    ("Eigen SE3",    "g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())"),
    ("Dense SE3",    "g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())"),
    ("Cholmod X",    "g2o.BlockSolverX(g2o.LinearSolverCholmodX())"),
    ("Eigen X",      "g2o.BlockSolverX(g2o.LinearSolverEigenX())"),
    ("Dense X",      "g2o.BlockSolverX(g2o.LinearSolverDenseX())"),
]
for name, code in solver_attempts:
    try:
        eval(code)
        print(f"   ✓ {name:15s}  {code}")
    except Exception as e:
        print(f"   ✗ {name:15s}  {e}")

print()
print("── Edge probe ───────────────────────────────")
for edge_name in ["EdgeSE3Expmap", "EdgeSE3", "EdgeSim3Expmap"]:
    try:
        getattr(g2o, edge_name)
        print(f"   ✓ g2o.{edge_name}")
    except AttributeError:
        print(f"   ✗ g2o.{edge_name}  (not available)")