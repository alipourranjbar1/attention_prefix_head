import itertools
import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from acceptance_block_labels import cached_boundary,verified_boundary,eligible_starts


class Labels(unittest.TestCase):
    def test_edges_and_first_rejection(self):
        self.assertEqual(verified_boundary([1,2,3],[9,2,3]),(0,'exact'))
        self.assertEqual(verified_boundary([1,2,3],[1,9,3]),(1,'exact'))
        self.assertEqual(verified_boundary([1,2,3],[1,2,3]),(3,'exact'))
        self.assertEqual(cached_boundary([1,2],[9,2],[1,2]),(None,'verify'))
        self.assertEqual(cached_boundary([1,2],[1,9],[1,2]),(2,'exact'))
        self.assertEqual(cached_boundary([9,2],[1,2],[1,2]),(0,'exact'))

    def test_eos_only_on_reachable_prefix(self):
        self.assertEqual(verified_boundary([1,2],[1,2],[2]),(None,'eos'))
        self.assertEqual(verified_boundary([9,2],[1,2],[2]),(0,'exact'))
        self.assertEqual(cached_boundary([1,3],[1,3],[1,2],[2]),(None,'eos'))

    def test_exhaustive_reference_divergence(self):
        cases=fallback=0
        # Different causal targets, arbitrary recorded sequences (not assumed greedy).
        for seed in range(40):
            rng=random.Random(seed)
            table={p:rng.randrange(3) for n in range(1,5)
                   for p in itertools.product(range(3),repeat=n)}
            for reference in itertools.product(range(3),repeat=3):
                prefix=(1,)
                cached=[table[prefix+reference[:j]] for j in range(3)]
                for proposed in itertools.product(range(3),repeat=3):
                    actual=[table[prefix+proposed[:j]] for j in range(3)]
                    for eos in ((),(2,)):
                        expected=verified_boundary(proposed,actual,eos)
                        observed=cached_boundary(proposed,reference,cached,eos)
                        if observed[1]=='verify':
                            fallback+=1
                            observed=verified_boundary(proposed,actual,eos)
                        self.assertEqual(observed,expected)
                        cases+=1
        self.assertGreater(fallback,0)
        print(f'Validated {cases:,} causal label cases, including {fallback:,} required fallbacks')

    def test_document_and_mask_boundaries(self):
        self.assertEqual(eligible_starts([1]*6,[0,0,0,1,1,1],3),[0,3])
        self.assertEqual(eligible_starts([1,1,0,1,1],[0]*5,3),[])
        self.assertEqual(eligible_starts([1]*4,[0,0,-1,-1],3),[])
        self.assertEqual(eligible_starts([0,1,1,1],[0]*4,3),[1])
        self.assertEqual(eligible_starts([1,1],[0,0],3),[])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): cached_boundary([],[],[])
        with self.assertRaises(ValueError): verified_boundary([1],[1,2])
        with self.assertRaises(ValueError): eligible_starts([1],[0,0],2)


if __name__=='__main__': unittest.main()
