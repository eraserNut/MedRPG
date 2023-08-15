from .trans_vg_ca import TransVG_ca


def build_model(args):
    if args.model_name == 'TransVG_ca':
        return TransVG_ca(args)
