import os
import yaml


class Config(object):
    def __init__(self, dpth):
        with open(dpth, "r", encoding="utf-8") as fp:
            conf_dict = yaml.load(fp, Loader=yaml.FullLoader)
            print("====config dict of ", dpth, "====")
            print(conf_dict)
            for key, val in conf_dict.items():
                setattr(self, key, val)

