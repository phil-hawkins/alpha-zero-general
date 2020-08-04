from absl import app, flags, logging
from absl.flags import FLAGS
import subprocess


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def git_record():
    git_msg = subprocess.check_output(['git', 'log', '-n 1 --pretty=format:%s HEAD'])
    return git_msg.decode("utf-8")


def config_rec():
    ignore_flags = [
        '?',
        'alsologtostderr',
        'help',
        'helpfull',
        'helpshort',
        'helpxml',
        'logtostderr',
        'log_dir',
        'only_check_args',
        'pdb_post_mortem',
        'profile_file',
        'run_with_pdb',
        'run_with_profiling',
        'showprefixforinfo',
        'stderrthreshold',
        'use_cprofile_for_profiling',
        'v',
        'verbosity',
        'op_conversion_fallback_to_while_loop',
        'test_random_seed',
        'test_srcdir',
        'test_tmpdir',
        'test_randomize_ordering_seed',
        'xml_output_file',
        'verbose',
        'graphic',
        'node_nums'
    ]
    crec = {flag:FLAGS[flag].value for flag in FLAGS if flag not in ignore_flags}
    crec["git_record"] = git_record()
    
    return crec
