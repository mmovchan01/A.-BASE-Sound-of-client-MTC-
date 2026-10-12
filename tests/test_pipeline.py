import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mtc_lib.common import encode, decode, macro_f1

class PipelineTest(unittest.TestCase):
    def test_labels(self):
        labels=np.array(['angry','neutral','positive','sad'])
        np.testing.assert_array_equal(decode(encode(labels)),labels)
        self.assertEqual(macro_f1(np.arange(4),np.arange(4)),1.)
        with self.assertRaises(ValueError): encode(['bad'])

    def test_train_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rng=np.random.default_rng(123)
            x=rng.normal(size=(160,12)).astype('float32')
            y=decode(np.arange(160)%4)
            np.savez(p/'train.npz',embeddings=x,targets=y)
            np.savez(p/'test.npz',embeddings=x[:7])
            (p/'cfg.json').write_text(json.dumps({'training':{'hidden':[8], 'epochs':2,'batch_size':32}}))
            subprocess.run([sys.executable,'train.py','--data',str(p/'train.npz'),'--out',str(p/'weights'),'--config',str(p/'cfg.json'),'--ensemble','1','--seed','123','--threads','1','--holdout','.2'],check=True,capture_output=True)
            subprocess.run([sys.executable,'predict.py','--test',str(p/'test.npz'),'--weights',str(p/'weights'),'--out',str(p/'submission_seed_123.npz')],check=True,capture_output=True)
            with np.load(p/'submission_seed_123.npz') as d:
                self.assertEqual(set(d.files),{'embeddings','targets'})
                np.testing.assert_array_equal(d['embeddings'],x[:7])
                self.assertEqual(d['targets'].shape,(7,))
                self.assertTrue(set(d['targets']).issubset(set(y)))
if __name__=='__main__': unittest.main()
