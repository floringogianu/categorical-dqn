from data_structures.ntuple_experience_replay import nTupleExperienceReplay
from data_structures.tensor_experience_replay import TensorExperienceReplay


class ExperienceReplay(object):
    @staticmethod
    def factory(cmdl, state_dims):
        type_name = cmdl.experience_replay

        if type_name == "nTupleExperienceReplay":
            return nTupleExperienceReplay(
                cmdl.replay_mem_size, cmdl.batch_size,
                cmdl.hist_len, cmdl.cuda
            )

        if type_name == "TensorExperienceReplay":
            if hasattr(cmdl, 'rescale_dims'):
                state_dims = (cmdl.rescale_dims, cmdl.rescale_dims)
            return TensorExperienceReplay(
                cmdl.replay_mem_size, cmdl.batch_size,
                cmdl.hist_len, state_dims, cmdl.cuda
            )
        assert 0, "Bad ExperienceReplay creation: " + type_name
