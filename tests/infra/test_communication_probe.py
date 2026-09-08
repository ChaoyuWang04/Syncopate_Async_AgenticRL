import unittest
from syncopate.infra import communication_probe as p

class CommunicationTests(unittest.TestCase):
    def test_nccl_version_metadata_is_allowed_and_labeled(self):
        result=p.validate_nccl_environment({'NCCL_VERSION':'2.28.3-1','NCCL_DEBUG':'INFO'})
        self.assertEqual(result['metadata'],{'NCCL_VERSION':'2.28.3-1'})
        self.assertFalse(result['metadata_is_runtime_version'])
    def test_nccl_algorithm_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'NCCL_ALGO'):
            p.validate_nccl_environment({'NCCL_VERSION':'2.28.3-1','NCCL_ALGO':'Ring'})
    def test_focused_rounds_balance_size_order(self):
        rounds=p.focused_schedule()
        self.assertEqual(len(rounds),3)
        self.assertTrue(all(len(row)==3 for row in rounds))
        self.assertEqual(len({row[0] for row in rounds}),3)
    def test_topodump_requires_own_output(self):
        from pathlib import Path
        p.validate_nccl_environment({'NCCL_TOPO_DUMP_FILE':'/tmp/myrun/topo.xml'},Path('/tmp/myrun'))
        with self.assertRaises(ValueError):
            p.validate_nccl_environment({'NCCL_TOPO_DUMP_FILE':'/tmp/other/topo.xml'},Path('/tmp/myrun'))
        with self.assertRaises(ValueError):
            p.validate_nccl_environment({'NCCL_PROTO':'LL128'},Path('/tmp/myrun'))
    def test_bytes_and_bus_convention(self):
        self.assertEqual(p.layout('all_gather_into_tensor', 1024)['algorithm_bytes'],2048)
        self.assertEqual(p.bandwidth('all_gather_into_tensor',1024,1e-6)['bus_GB_s'],1.024)
        self.assertEqual(p.bandwidth('all_reduce',1024,1e-6)['bus_GB_s'],1.024)
        with self.assertRaises(ValueError):p.layout('all_reduce',3)
    def test_matrix_keeps_shape_sources_and_neighbor_scope(self):
        m=p.message_matrix([2048,4096],8)
        self.assertTrue(any(x['block_bytes']==2*2048*4096 for x in m))
        self.assertTrue(any(x['block_bytes']==2*8*4096 for x in m))
        self.assertTrue(all(x['workload_frequency']=='unmeasured' for x in m))
        self.assertTrue(any(x['block_bytes']%16==2 for x in m))
    def test_reject_empty_or_failed_rank_results(self):
        with self.assertRaises(ValueError):p.aggregate([])
        with self.assertRaises(ValueError):p.aggregate([{'rank':0,'ok':False},{'rank':1,'ok':True}])
    def test_neighbor_screen_requires_effect_and_noise(self):
        rows=[{'operation':'all_reduce','block_bytes':16,'timing':{'seconds_per_call_summary':{'median':1.,'sample_stdev':.01}}},
              {'operation':'all_reduce','block_bytes':18,'timing':{'seconds_per_call_summary':{'median':1.3,'sample_stdev':.02}}}]
        messages=[{'block_bytes':18,'sources':[{'source':'test','delta_bytes':2,'source_bytes':16}]}]
        self.assertTrue(p.screen_neighbors(rows,messages)[0]['flag'])
        rows[1]['timing']['seconds_per_call_summary']['sample_stdev']=.2
        self.assertFalse(p.screen_neighbors(rows,messages)[0]['flag'])
    def test_source_shape_invalid(self):
        for shape in ([0,2],[2],[2,-4]):
            with self.assertRaises(ValueError):p.message_matrix(shape,8)

if __name__=='__main__':unittest.main()
