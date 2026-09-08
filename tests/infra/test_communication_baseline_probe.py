import pytest
from syncopate.infra import communication_baseline_probe as p


def test_rounding_is_upstream_16_byte_per_rank():
    center=16<<20
    assert p.native_layout('all_reduce',center+2)['actual_bytes']==center+2
    assert p.native_layout('all_gather',center-2)['actual_bytes']==center-32
    assert p.native_layout('reduce_scatter',center+2)['actual_bytes']==center
    assert len(p.cases())==7


@pytest.mark.parametrize('size',[0,-2,3,True])
def test_invalid_sizes(size):
    with pytest.raises(ValueError):p.native_layout('all_reduce',size)


def test_parser_preserves_both_modes_and_rejects_wrong():
    text='16777216 8388608 bfloat16 sum -1 10.5 1597.83 1597.83 0 11 1525.2 1525.2 0\n'
    row=p.parse_native(text)[0]
    assert row['in_place']['seconds']==11e-6
    assert row['out_of_place']['wrong']==0
    with pytest.raises(ValueError):p.parse_native(text.replace('1525.2 0','1525.2 1'))
    with pytest.raises(ValueError):p.parse_native('# no rows')
    with pytest.raises(ValueError):p.parse_native(text.replace('10.5','nan'))


def test_native_command_fixes_layout_and_max_rank():
    cmd=p.native_command('/tmp/build','all_reduce',16777216,256,3)
    assert cmd[:4]==['mpirun','--allow-run-as-root','--bind-to','none']
    for flag,value in [('-np','2'),('-g','1'),('-t','1'),('-a','3'),('-m','1'),('-c','1')]:
        assert cmd[cmd.index(flag)+1]==value


def test_calibration_bounds():
    assert p.calibrate_iterations(.001)==128
    assert p.calibrate_iterations(.000001)==4096
    with pytest.raises(ValueError):p.calibrate_iterations(0)


def test_noise_gate_does_not_turn_engineering_success_into_baseline():
    assert p.sample_summary([1.,1.01,1.02])['within_allocation_stable']
    assert not p.sample_summary([1.,1.2,1.])['within_allocation_stable']
    with pytest.raises(ValueError):p.sample_summary([1.,2.])


def test_torchrun_uses_canonical_module_when_parent_runs_as_main():
    cmd=p.torch_command('/out','/cpu')
    assert cmd[-8:] == ['-m','syncopate.infra.communication_baseline_probe','--mode','rank','--out','/out','--preflight','/cpu']


def test_bootstrap_environment_is_shared_and_explicit():
    source={'PATH':'/bin','PMIX_MCA_gds':'hash'}
    env=p.execution_environment(source)
    assert env['PMIX_MCA_gds']=='hash'
    assert env['OMP_NUM_THREADS']=='1'
    assert source=={'PATH':'/bin','PMIX_MCA_gds':'hash'}
    with pytest.raises(ValueError):p.execution_environment({'PMIX_MCA_gds':'shmem'})


def test_mpi_health_requires_two_ranks_and_collective_result():
    good='MPI_HEALTH rank=0 size=2 sum=3\nMPI_HEALTH rank=1 size=2 sum=3\n'
    assert p.parse_mpi_health(good)==[{'rank':0,'size':2,'sum':3},{'rank':1,'size':2,'sum':3}]
    for bad in ['',good.splitlines()[0],good.replace('sum=3','sum=2'),good.replace('rank=1','rank=0')]:
        with pytest.raises(ValueError):p.parse_mpi_health(bad)


def test_teardown_releases_graphs_before_group_and_publishes_after(tmp_path):
    events=[]
    class FakeGraph:
        def __del__(self):events.append('release')
    graphs=[FakeGraph(),FakeGraph()]
    class Cuda:
        @staticmethod
        def synchronize():events.append('sync')
    class Torch:cuda=Cuda()
    class Dist:
        @staticmethod
        def destroy_process_group():
            state=__import__('json').loads((tmp_path/'rank-0.json').read_text())
            assert state['ok'] is False and state['phase']=='teardown_started'
            assert events==['sync','release','release']
            events.append('destroy')
    result=p.finish_rank(Torch(),Dist(),graphs,tmp_path,{'rank':0,'cases':[{'measured':True}]})
    assert graphs==[] and events[-1]=='destroy'
    assert result['ok'] is True and result['phase']=='complete'


def test_teardown_failure_keeps_measurements_but_never_ok(tmp_path):
    class Cuda:
        @staticmethod
        def synchronize():pass
    class Torch:cuda=Cuda()
    class Dist:
        @staticmethod
        def destroy_process_group():raise RuntimeError('teardown failed')
    with pytest.raises(RuntimeError):p.finish_rank(Torch(),Dist(),[],tmp_path,{'rank':1,'cases':[1]})
    state=__import__('json').loads((tmp_path/'rank-1.json').read_text())
    assert state=={'rank':1,'cases':[1],'ok':False,'phase':'teardown_started'}


def test_native_graph_schedule_balances_all_arm_positions():
    schedule=p.native_graph_schedule()
    assert len(schedule)==3
    for position in range(3):assert {row[position] for row in schedule}==set(p.GRAPH_ARMS)


def test_native_graph_command_preserves_exact_operation_count():
    for arm in p.GRAPH_ARMS:
        cmd=p.native_graph_command('/build','all_reduce',arm)
        assert cmd[cmd.index('-n')+1]=='1024'
        assert cmd[cmd.index('-m')+1]=='1'
        assert cmd[cmd.index('-N')+1]=='1'
        assert cmd[cmd.index('-G')+1]==('0' if arm=='native_eager' else '1')
    with pytest.raises(ValueError):p.native_graph_command('/build','all_reduce','invented')


def test_native_graph_environment_only_allows_registered_override():
    env=p.native_graph_environment({},'native_graph_mixing_off')
    assert env['NCCL_GRAPH_MIXING_SUPPORT']=='0'
    assert 'NCCL_GRAPH_MIXING_SUPPORT' not in p.native_graph_environment({},'native_graph')
    with pytest.raises(ValueError):p.native_graph_environment({'NCCL_ALGO':'Ring'},'native_eager')


def test_native_graph_requires_consumed_flag_and_mixing_evidence():
    row='16777216 8388608 bfloat16 sum -1 52 322 322 0 53 316 316 0\n'
    log='# nThread 1 nGpus 1 graph: 1\nNCCL_GRAPH_MIXING_SUPPORT set by environment to 0\n'+row
    assert p.verify_native_graph_log(log,'all_reduce','native_graph_mixing_off',log)[0]['count']==8388608
    with pytest.raises(ValueError):p.verify_native_graph_log(row,'all_reduce','native_graph')
    with pytest.raises(ValueError):p.verify_native_graph_log('# graph: 1\n'+row,'all_reduce','native_graph_mixing_off')


def test_native_diagnostics_require_two_distinct_owned_ranks(tmp_path):
    prefix='window-0-native_eager-all_reduce-nccl-'
    for rank in (0,1):
        (tmp_path/f'{prefix}host-{100+rank}.log').write_text(f'NCCL INFO comm 0xabc rank {rank} nranks 2 cudaDev {rank} - Init COMPLETE\n')
    (tmp_path/'unrelated.log').write_text('not this invocation')
    evidence=p.read_native_diagnostics(tmp_path,prefix)
    assert {f['rank'] for f in evidence['files']}=={0,1}
    (tmp_path/f'{prefix}host-101.log').write_text('rank 0 nranks 2\n')
    with pytest.raises(ValueError):p.read_native_diagnostics(tmp_path,prefix)


def test_benchmark_parser_and_mixing_evidence_are_separate():
    row='16777216 8388608 bfloat16 sum -1 52 322 322 0 53 316 316 0\n'
    clean='# graph: 1\n'+row
    diagnostic='NCCL_GRAPH_MIXING_SUPPORT set by environment to 0\n'
    assert p.verify_native_graph_log(clean,'all_reduce','native_graph_mixing_off',diagnostic)[0]['count']==8388608
    with pytest.raises(ValueError):p.verify_native_graph_log(clean.replace('sum -1','sum -1NCCL INFO'),'all_reduce','native_graph_mixing_off',diagnostic)


def test_direct_call_supplies_exact_one_tensor_comm_and_stream():
    observed={}
    class Nccl:
        SUM=0
        @staticmethod
        def all_reduce(inputs,outputs,op,streams,comms):observed.update(inputs=inputs,outputs=outputs,op=op,streams=streams,comms=comms)
    tensor=object();stream=object();comm=object()
    p.direct_all_reduce(Nccl,tensor,stream,comm)
    assert observed==dict(inputs=[tensor],outputs=[tensor],op=0,streams=[stream],comms=[comm])


def test_direct_command_uses_fixed_rank_entry():
    assert p.torch_command('/out','/cpu',direct=True)[-7:] == ['syncopate.infra.communication_baseline_probe','--mode','direct-rank','--out','/out','--preflight','/cpu']


def test_graph_summary_requires_kernel_nodes_and_preserves_types():
    data={'nodes':[{'node_type':'kernel','kernel_name':'ncclKernel','dependencies':[]},{'node_type':'event_record','kernel_name':None,'dependencies':[0]}]}
    result=p.graph_summary(data)
    assert result['node_types']=={'kernel':1,'event_record':1}
    assert result['kernel_names']=={'ncclKernel':1}
    with pytest.raises(ValueError):p.graph_summary({'nodes':[]})


def test_all_local_run_calls_bind_to_shared_runner_signature():
    import ast
    import inspect
    tree=ast.parse(inspect.getsource(p))
    signature=inspect.signature(p.run)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='run']
    assert calls
    for call in calls:
        assert not any(isinstance(arg,ast.Starred) for arg in call.args)
        assert all(keyword.arg is not None for keyword in call.keywords)
        try:
            signature.bind(*[object() for _ in call.args],**{keyword.arg:object() for keyword in call.keywords})
        except TypeError as error:
            raise AssertionError(f'run call at line {call.lineno}: {error}') from error
