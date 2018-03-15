# Global config


class Config(object):
    # global options
    data_format = 'NCHW'    # DO NOT CHANGE THIS
    SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"
    sn = True   # Whether uses Spectral Normalization(https://arxiv.org/abs/1802.05957)
    proj_d = False    # Whether uses projection discriminator(https://arxiv.org/abs/1802.05637)
    wgan = False    # WGAN or DRAGAN(only effective if sn is False)
    pre_calculated_dist_map = True    # Whether calculate distance maps on the fly

    @staticmethod
    def set_from_dict(d):
        assert type(d) is dict
        for k, v in d.items():
            setattr(Config, k, v)
