# Global config


class Config(object):
    # global options
    data_format = 'NCHW'
    sn = False
    proj_d = False
    wgan = False
    SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"
    pre_calculated_dist_map = True

    @staticmethod
    def set_from_dict(d):
        assert type(d) is dict
        for k, v in d.items():
            setattr(Config, k, v)
