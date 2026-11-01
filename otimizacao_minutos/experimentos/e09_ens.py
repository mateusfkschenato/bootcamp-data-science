from modelos import *
ensemble("ENS RFSsemcw+HGBtodas", ["RF S sem cw","HGB todas"])
ensemble("ENS RFSsemcw+HGBl2=5", ["RF S sem cw","HGB todas l2=5"])
ensemble("ENS RFtodas+HGBl2=5", ["RF todas sem cw mf.2","HGB todas l2=5"])
ensemble("ENS 3", ["RF S sem cw","RF todas sem cw mf.2","HGB todas l2=5"])
ensemble("ENS RFS(cw)+HGB logit", ["RF S","HGB todas l2=5"], modo="logit")
ensemble("ENS 4", ["RF S sem cw","RF todas sem cw mf.2","HGB todas l2=5","HGB orig84+H+outras"])
