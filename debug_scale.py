import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

debug_code = """        # IQR filtering for robustness
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_ratios = ratios[(ratios >= lower) & (ratios <= upper)]
        
        if len(filtered_ratios) < 3:
            ratio = np.median(ratios)
        else:
            ratio = np.median(filtered_ratios)
        
        if self.frame_id % 100 == 0:
            print(f"  [Scale Debug] Frame {self.frame_id} | median_ratio={ratio:.3f} | points={len(mps_ref)}")
"""

code = code.replace('# IQR filtering for robustness', debug_code)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)
