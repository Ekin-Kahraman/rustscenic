import unittest,sys
from pathlib import Path
import numpy as np,pandas as pd
from corrected_evaluation import bh_adjust,calibrated_labels,pooled_log_effect
from analysis_paths import PACKAGE
sys.path.insert(0,str(PACKAGE))
from metrics import expected_hits,score_rank

class EvaluationRegression(unittest.TestCase):
 def test_pool_before_log_removes_constructed_small_pool_bias(self):
  # The small pools alternate absent/detected expression; their pooled CPM is identical to controls.
  a=np.array([[0.,0.],[2.,20.]]);b=np.array([[1.,10.],[1.,10.]])
  biased=(np.log2(1+a)-np.log2(1+b)).mean(0)
  self.assertTrue((biased<0).all());np.testing.assert_allclose(pooled_log_effect(a,b,[1,1]),0,atol=1e-15)
 def test_constant_expression_is_invariant_to_pool_size(self):
  a=np.array([[3.,7.],[3.,7.],[3.,7.]])
  np.testing.assert_allclose(pooled_log_effect(a,a,[2,40,100]),0)
 def test_bh_matches_known_monotone_adjustment(self):
  np.testing.assert_allclose(bh_adjust([.01,.04,.03,.002]),[.02,.04,.04,.008])
  with self.assertRaises(ValueError):bh_adjust([np.nan])
 def test_qc_missingness_effect_threshold_and_self_exclusion(self):
  f=pd.DataFrame({'grna_target':['A']*4,'response_id':['B','C','D','E'],'p_value':[.001,.9,np.nan,.000001],'pass_qc':[True,True,True,False],'log_2_fold_change':[1.,2.,1.,1.]})
  a=calibrated_labels(f);self.assertEqual(a.eligible.tolist(),[True,True,False,False]);self.assertEqual(a.response.tolist(),[True,False,False,False]);self.assertAlmostEqual(a.q_value.iloc[0],.002)
  f.loc[0,'response_id']='A'
  with self.assertRaises(ValueError):calibrated_labels(f)
 def test_absent_predictions_never_gain_emitted_hits(self):
  y=np.array([True,False,True,False]);s=np.zeros(4);m=score_rank(y,s,0)
  self.assertEqual(m['emitted_hits_at_50'],0);self.assertEqual(m['predictions_at_50'],0);self.assertEqual(m['average_precision'],.5)
  self.assertEqual(expected_hits(y,s,1),.5)
 def test_tf_gene_permutations_preserve_bh(self):
  p=np.array([.001,.005,.4,.9,.006]);ix=np.array([4,2,0,3,1]);np.testing.assert_allclose(bh_adjust(p)[ix],bh_adjust(p[ix]))
if __name__=='__main__':unittest.main()
