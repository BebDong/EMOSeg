from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS


@HOOKS.register_module()
class RunnerInfoHook(Hook):
    def __init__(self, partition='decode_head'):
        assert partition in ('segmentor', 'backbone', 'neck', 'decode_head')
        self.partition = partition

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        cur_iter = runner.iter
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        model.decode_head.set_cur_iter(cur_iter)
