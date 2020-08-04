from absl import app, flags, logging
from absl.flags import FLAGS
import subprocess

def git_record():
    git_msg = subprocess.check_output(['git', 'log', '-n 1 --pretty=format:%s HEAD'])
    return git_msg.decode("utf-8")

def flags_text():
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
        'verbosity'
    ]
    flag_lst = ["    {} : {}".format(flag, FLAGS[flag].value) for flag in FLAGS if flag not in ignore_flags]
    
    return "\n".join(flag_lst)

def run_settings(config_dict):
    config_text = "\n".join(["    {} : {}".format(k,v) for k,v in config_dict.items()])
    return "Flags:\n{}\nConfig:\n{}\nGit\n{}".format(flags_text(), config_text, git_record())