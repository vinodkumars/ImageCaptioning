import pickle
import os

pkl_path = 'hypo7.pkl'
[hypo, ref] = pickle.load(open(pkl_path, 'rb'))

filename = 'hypo.txt'
if os.path.exists(filename):
  os.remove(filename)
with open(filename, 'w') as f:
  for k,v in hypo.iteritems():
    #f.write(str(k))
    #f.write('\t')
    f.write(v[0].lower().strip())
    f.write('\n')
    
ref0 = 'ref0.txt'
ref1 = 'ref1.txt'
ref2 = 'ref2.txt'
ref3 = 'ref3.txt'
ref4 = 'ref4.txt'
ref_list = [ref0, ref1, ref2, ref3, ref4]
for r in ref_list:
  if os.path.exists(r):
    os.remove(r)
with open(ref0, 'w') as ref0_f, open(ref1, 'w') as ref1_f, open(ref2, 'w') as ref2_f, open(ref3, 'w') as ref3_f, open(ref4, 'w') as ref4_f:
  ref_f = [ref0_f, ref1_f, ref2_f, ref3_f, ref4_f]
  for k,v_arr in ref.iteritems():
    for v, f in zip(v_arr, ref_f):
      #f.write(str(k))
      #f.write('\t')
      f.write(v.lower().strip())
      f.write('\n')
